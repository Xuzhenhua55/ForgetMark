import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


"""
ForgetMark: Uncertainty-Driven Key–Value Selection
--------------------------------------------------

This script implements the fingerprint construction stage described in the paper:
- For each Key (prompt), sample M independent continuations from the target model.
- At each generation step, obtain the model's logits and compute the probability of
  the actually generated token; aggregate per-token log-probabilities to get sequence NLL.
- Compute per-Key predictive uncertainty U_i as the average sequence NLL across M samples:

    U_i = - (1/M) * sum_j sum_t log p_{i,j}^{(t)}

- Select the N Keys with the smallest U_i (lowest uncertainty / highest determinacy).
- For each selected Key, keep the continuation with the minimal NLL as its Value.

Inputs
------
- JSON file of objects with a "question" field (treated as a Key).

Outputs
-------
- JSON file containing full sampling details per Key (texts, token ids, per-token log probs, NLL),
  each Key's U_i, the selected Key indices, and the final fingerprint set F = {(k_i, v_i)}.

Notes
-----
- We rely on HuggingFace generation with return_dict_in_generate=True and output_scores=True
  to extract step-wise logits. We then compute log-softmax to gather token log-probabilities.
- Chat models: if the tokenizer defines a chat template, we apply it; otherwise we use the raw text.
"""


@dataclass
class SampleResult:
    text: str
    token_ids: List[int]
    log_probs: List[float]  # per-token log probability

    @property
    def nll(self) -> float:
        # Negative log-likelihood = - sum(log_probs)
        return -float(sum(self.log_probs))


@dataclass
class KeyResult:
    key: str
    samples: List[SampleResult]
    U: float  # predictive uncertainty as defined above

    def best_sample(self) -> SampleResult:
        # Minimal NLL sample is the most likely continuation
        return min(self.samples, key=lambda s: s.nll)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_input_text(tokenizer: AutoTokenizer, user_text: str, system_prompt: Optional[str]) -> str:
    """Build the input string using chat template if available.

    If the tokenizer provides an `apply_chat_template`, we use a System+User two-turn chat.
    Otherwise, return a simple concatenation.
    """
    if hasattr(tokenizer, "apply_chat_template") and callable(tokenizer.apply_chat_template):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_text})
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # Fallback: plain text prompt
    sys_part = f"System: {system_prompt}\n" if system_prompt else ""
    return f"{sys_part}User: {user_text}\nAssistant:"


@torch.no_grad()
def sample_with_logprobs(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt_text: str,
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_p: float = 0.9,
    seed: Optional[int] = None,
) -> SampleResult:
    """Generate one continuation and record per-token log probabilities of generated tokens.

    We call `generate` with `output_scores=True` and `return_dict_in_generate=True`.
    The returned `scores` is a list of logits per generated step. We take log-softmax and
    gather the log-probability of the actually generated token at each step.
    """
    if seed is not None:
        set_seed(seed)

    inputs = tokenizer([prompt_text], return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    gen_out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        output_scores=True,
        return_dict_in_generate=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # generated sequences shape: [batch=1, seq_len]
    sequences: torch.Tensor = gen_out.sequences  # includes the input prompt prefix
    scores: List[torch.Tensor] = gen_out.scores  # list length T, each [batch, vocab]

    # Remove the input prefix to get only generated tokens
    input_len = inputs["input_ids"].shape[1]
    gen_token_ids = sequences[0, input_len:].tolist()

    log_probs: List[float] = []
    for t, token_id in enumerate(gen_token_ids):
        # scores[t] is the logits BEFORE softmax for step t
        step_logits = scores[t][0]  # [vocab]
        step_logprobs = F.log_softmax(step_logits, dim=-1)
        log_probs.append(float(step_logprobs[token_id].item()))

    text = tokenizer.decode(gen_token_ids, skip_special_tokens=True).strip()
    return SampleResult(text=text, token_ids=gen_token_ids, log_probs=log_probs)


def compute_predictive_uncertainty(samples: List[SampleResult]) -> float:
    """Compute U = - (1/M) * sum_j sum_t log p^{(t)} for M samples.

    Note: This matches the paper's definition and does not normalize by sequence length.
    """
    if not samples:
        return float("inf")
    total_log_prob_sum = 0.0
    for s in samples:
        total_log_prob_sum += sum(s.log_probs)
    M = len(samples)
    U = - (total_log_prob_sum / M)
    return float(U)


def load_model_and_tokenizer(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    return model, tokenizer


def process_keys(
    keys: List[str],
    model_path: str,
    M: int,
    N: int,
    system_prompt: Optional[str] = "You are a helpful assistant.",
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_p: float = 0.9,
    base_seed: int = 42,
) -> Dict[str, Any]:
    """Run the uncertainty-driven selection pipeline over a list of Keys.

    Returns a dictionary ready to be serialized to JSON with:
      - per-key samples and U
      - selected indices
      - the final fingerprint set (key, value)
    """
    model, tokenizer = load_model_and_tokenizer(model_path)

    key_results: List[KeyResult] = []
    for i, key in enumerate(keys):
        prompt = build_input_text(tokenizer, key, system_prompt)

        samples: List[SampleResult] = []
        for j in range(M):
            seed = base_seed + j  # different seed per sample
            s = sample_with_logprobs(
                model,
                tokenizer,
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                seed=seed,
            )
            samples.append(s)

        U = compute_predictive_uncertainty(samples)
        key_results.append(KeyResult(key=key, samples=samples, U=U))

    # Select N keys with the smallest U
    idx_sorted = sorted(range(len(key_results)), key=lambda idx: key_results[idx].U)
    selected_indices = idx_sorted[: min(N, len(idx_sorted))]

    fingerprint_set: List[Dict[str, Any]] = []
    for idx in selected_indices:
        kr = key_results[idx]
        best = kr.best_sample()
        fingerprint_set.append(
            {
                "key": kr.key,
                "value": best.text,
                "U": kr.U,
                "value_nll": best.nll,
            }
        )

    # Build JSON-serializable structure
    results_json: Dict[str, Any] = {
        "model": model_path,
        "M": M,
        "N": N,
        "system_prompt": system_prompt,
        "keys_count": len(keys),
        "selected_indices": selected_indices,
        "fingerprint_set": fingerprint_set,
        "per_key": [],
    }

    for kr in key_results:
        per_key_entry = {
            "key": kr.key,
            "U": kr.U,
            "samples": [
                {
                    "text": s.text,
                    "token_ids": s.token_ids,
                    "log_probs": s.log_probs,
                    "nll": s.nll,
                }
                for s in kr.samples
            ],
        }
        results_json["per_key"].append(per_key_entry)

    return results_json


def read_keys_from_json(input_json_path: str) -> List[str]:
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    keys: List[str] = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and "question" in item and isinstance(item["question"], str):
                keys.append(item["question"].strip())
            elif isinstance(item, str):
                keys.append(item.strip())
    elif isinstance(data, dict) and "questions" in data and isinstance(data["questions"], list):
        for q in data["questions"]:
            if isinstance(q, str):
                keys.append(q.strip())
            elif isinstance(q, dict) and "question" in q:
                keys.append(str(q["question"]).strip())
    return keys


def write_json(obj: Dict[str, Any], output_json_path: str) -> None:
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ForgetMark: Uncertainty-driven Key–Value selection for fingerprint construction",
    )
    parser.add_argument("--model_path", type=str, required=True, help="HF model path or local directory")
    parser.add_argument("--input_json", type=str, required=True, help="Input JSON with Keys (field: 'question')")
    parser.add_argument("--output_json", type=str, required=True, help="Output JSON path for results")
    parser.add_argument("--M", type=int, default=3, help="Number of samples per Key")
    parser.add_argument("--N", type=int, default=100, help="Number of Keys to select (lowest U)")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Max new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p nucleus sampling")
    parser.add_argument("--system_prompt", type=str, default="You are a helpful assistant.", help="System prompt")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    keys = read_keys_from_json(args.input_json)
    if not keys:
        raise ValueError("No keys were found in the input JSON. Expected a list of {'question': str} or ['...'].")

    results = process_keys(
        keys=keys,
        model_path=args.model_path,
        M=args.M,
        N=args.N,
        system_prompt=args.system_prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        base_seed=args.seed,
    )

    write_json(results, args.output_json)
    print(f"Done. Selected {len(results['fingerprint_set'])} Keys. Output saved to: {args.output_json}")


if __name__ == "__main__":
    main()

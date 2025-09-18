#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate a FULL fused HF model (no LoRA) on a QA dataset (forget split),
computing:
- forget_Q_A_Prob.mean (average probability)
- forget_Q_A_ROUGE(rougeL_recall).mean
- counts below probability threshold (FSR_prob)
  - default threshold: 0.001 (configurable via --prob-threshold)
- optionally, counts below ROUGE threshold (FSR_rouge) if --rouge-threshold is set

Usage 1: Single model
CUDA_VISIBLE_DEVICES=0 python fsr_prob_rouge.py \
  --model \
    "..." \
  --dataset \
    "..." \
  --batch-size 16 \
  --max-length 512 \
  --gpu-ids 0 \
  --prob-threshold 0.001 \
  --auto-out true
"""
import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from omegaconf import OmegaConf

# add src/ to sys.path (expects open-unlearning style package layout)
CUR_DIR = Path(__file__).resolve().parent
SRC_DIR = (CUR_DIR / ".." / "src").resolve()
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.qa import QADataset
from data.collators import DataCollatorForSupervisedDataset
from evals.metrics.utils import (
    run_batchwise_evals,
    evaluate_probability,
    eval_text_similarity,
)


def str2bool(v: str) -> bool:
    return v.lower() in ("true", "1", "yes", "y", "t")


def get_device_dtype(dtype_str: str, gpu_ids: List[int]) -> Tuple[torch.device, torch.dtype]:
    """
    Determine device and dtype based on provided GPU IDs and dtype string.
    If multiple GPUs are provided, we use device_map="auto" and let HF shard.
    """
    if torch.cuda.is_available() and len(gpu_ids) > 0:
        available_gpus = list(range(torch.cuda.device_count()))
        for gid in gpu_ids:
            if gid not in available_gpus:
                raise ValueError(f"GPU ID {gid} is not available. Available: {available_gpus}")
        main_gpu = gpu_ids[0]
        torch.cuda.set_device(main_gpu)
        device = torch.device(f"cuda:{main_gpu}")
    else:
        device = torch.device("cpu")

    if dtype_str == "bfloat16":
        dtype = torch.bfloat16
    elif dtype_str == "float16":
        dtype = torch.float16
    else:
        dtype = torch.float32
    return device, dtype


def load_full_model(model_path: str, dtype: torch.dtype, gpu_ids: List[int]):
    # device_map setup
    device_map = "auto"
    if torch.cuda.is_available() and len(gpu_ids) == 1:
        device_map = {"": f"cuda:{gpu_ids[0]}"}
    elif not torch.cuda.is_available():
        device_map = {"": "cpu"}

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map=device_map,
        trust_remote_code=True,
    )
    model.eval()
    return model


def load_tokenizer(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def get_template_args_for_model(tokenizer):
    """Return template args depending on whether tokenizer has a chat template."""
    has_chat_template = getattr(tokenizer, "chat_template", None) is not None and hasattr(tokenizer, "apply_chat_template")
    if has_chat_template:
        return dict(
            apply_chat_template=True,
            system_prompt="You are a helpful assistant.",
        )
    # Manual fallback: prefer a Mistral-style prompt if using a Mistral tokenizer; otherwise use a simple User/Assistant style
    name = str(getattr(tokenizer, "name_or_path", "")).lower()
    if "mistral" in name or "llama" in name:
        # Mistral/LLaMA-style instruction blocks using [INST] ... [/INST] and optional <<SYS>> ... <</SYS>>
        return dict(
            apply_chat_template=False,
            system_prompt_with_special_tokens="<<SYS>>\nYou are a helpful assistant.\n<</SYS>>\n",
            user_start_tag="[INST] ",
            user_end_tag=" ",
            asst_start_tag=" [/INST] ",
            asst_end_tag="\n",
        )
    # Fallback manual-tag template for base models without chat template
    return dict(
        apply_chat_template=False,
        system_prompt_with_special_tokens=None,
        user_start_tag="User: ",
        user_end_tag="\n",
        asst_start_tag="Assistant: ",
        asst_end_tag="\n",
    )


def build_datasets(tokenizer, dataset_path: str, max_length: int, template_args: dict):
    # probability: predict_with_generate=False (compute loss on answer tokens)
    ds_prob = QADataset(
        hf_args={"path": dataset_path, "split": "train"},
        template_args=template_args,
        tokenizer=tokenizer,
        question_key="question",
        answer_key="answer",
        max_length=max_length,
        predict_with_generate=False,
    )

    # rouge: predict_with_generate=True (generate from prompt-only)
    ds_rouge = QADataset(
        hf_args={"path": dataset_path, "split": "train"},
        template_args=template_args,
        tokenizer=tokenizer,
        question_key="question",
        answer_key="answer",
        max_length=max_length,
        predict_with_generate=True,
    )
    return ds_prob, ds_rouge


def get_dataloaders(ds_prob, ds_rouge, tokenizer, batch_size: int):
    from torch.utils.data import DataLoader
    collator_prob = DataCollatorForSupervisedDataset(tokenizer=tokenizer, padding_side="right", index="index")
    collator_rouge = DataCollatorForSupervisedDataset(tokenizer=tokenizer, padding_side="left", index="index")
    dl_prob = DataLoader(ds_prob, batch_size=batch_size, collate_fn=collator_prob)
    dl_rouge = DataLoader(ds_rouge, batch_size=batch_size, collate_fn=collator_rouge)
    return dl_prob, dl_rouge


def _auto_out_path_from_model(model_path: str, out_base: str) -> str:
    """Compose output path under out_base based on model_path.
    Expected structure: .../mistral_merges/<strategy>/<model_name>/<ratio>
    Fallbacks to using the last component as ratio if not strictly matched.
    Output file: out_base/<strategy>/<ratio_underscore>_prob_rouge.jsonl
    """
    p = Path(model_path).resolve()
    parts = p.parts
    strategy = None
    ratio = p.name
    try:
        idx = parts.index("mistral_merges")
        if idx + 1 < len(parts):
            strategy = parts[idx + 1]
        # ratio is ideally the last segment after model name
        if idx + 3 < len(parts):
            ratio = parts[idx + 3]
    except ValueError:
        # leave defaults
        pass

    if strategy is None:
        strategy = "unknown"

    ratio_us = str(ratio).replace("-", "_")
    out_dir = Path(out_base) / strategy
    out_dir.mkdir(parents=True, exist_ok=True)
    return str(out_dir / f"{ratio_us}_prob_rouge.jsonl")


def _discover_ratio_dirs(strategy_dir: Path) -> list:
    """Return a list of immediate child directory names in strategy_dir/model_name."""
    if not strategy_dir.exists():
        return []
    subdirs = [d.name for d in strategy_dir.iterdir() if d.is_dir()]
    return subdirs


def evaluate_and_save(
    model_path: str,
    dataset_path: str,
    batch_size: int,
    max_length: int,
    dtype_str: str,
    max_new_tokens: int,
    do_sample: bool,
    top_p: float,
    temperature: float,
    gpu_ids: List[int],
    out_path: str,
    stopwords: list = None,
    prob_threshold: float = 1e-3,
    rouge_threshold: float = None,
):
    # Ensure output directory
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # device + dtype (also sets CUDA device)
    device, dtype = get_device_dtype(dtype_str, gpu_ids)

    # Load model/tokenizer
    model = load_full_model(model_path, dtype, gpu_ids)
    tokenizer = load_tokenizer(model_path)

    # Templates
    template_args = get_template_args_for_model(tokenizer)

    # Datasets / loaders
    ds_prob, ds_rouge = build_datasets(tokenizer, dataset_path, max_length, template_args)
    dl_prob, dl_rouge = get_dataloaders(ds_prob, ds_rouge, tokenizer, batch_size)

    generation_args = {
        "do_sample": do_sample,
        "top_p": top_p,
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "use_cache": True,
    }
    if stopwords:
        generation_args["stopwords"] = list(stopwords)
    generation_cfg = OmegaConf.create(generation_args)

    # Probability metric
    with torch.no_grad():
        prob_by_index = run_batchwise_evals(
            model,
            dl_prob,
            evaluate_probability,
            batch_eval_fn_args={},
            eval_msg="Calculating loss (forget_Q_A_Prob)",
        )

    # ROUGE metric (generation)
    with torch.no_grad():
        rouge_by_index = run_batchwise_evals(
            model,
            dl_rouge,
            eval_text_similarity,
            batch_eval_fn_args={"tokenizer": tokenizer, "generation_args": generation_cfg},
            eval_msg="Calculating text similarity (forget_Q_A_ROUGE)",
        )

    # Save per-sample outputs
    output_lines = []
    for i in range(len(ds_prob)):
        rec = ds_prob.data[i]
        idx = int(rec.get("index", i))
        question = rec.get("question", None)
        answer = rec.get("answer", None)
        prob_entry = prob_by_index.get(idx, None)
        rouge_entry = rouge_by_index.get(idx, None)
        out_obj = {
            "index": idx,
            "question": question,
            "answer": answer,
            "forget_Q_A_Prob": prob_entry,
            "forget_Q_A_ROUGE": rouge_entry,
            "used_gpus": gpu_ids,
        }
        output_lines.append(out_obj)

    with open(out_path, "w", encoding="utf-8") as f:
        for obj in output_lines:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    # Aggregates
    import numpy as np
    probs = [v["prob"] for v in prob_by_index.values() if v is not None and v.get("prob") is not None]
    rouge_vals = [v.get("rougeL_recall", None) for v in rouge_by_index.values() if v is not None]
    rouge_vals = [x for x in rouge_vals if x is not None]

    total = len(probs)
    prob_mean = float(np.mean(probs)) if total else float("nan")
    rouge_mean = float(np.mean(rouge_vals)) if len(rouge_vals) else float("nan")

    # Threshold-based counts
    prob_lt_01 = int(np.sum(np.array(probs) < 0.1)) if total else 0
    prob_lt_thr = int(np.sum(np.array(probs) < float(prob_threshold))) if total else 0
    prob_lt_thr_rate = float(prob_lt_thr / total) if total else float("nan")

    rouge_lt_thr = None
    rouge_lt_thr_rate = None
    if rouge_threshold is not None and len(rouge_vals):
        rouge_lt_thr = int(np.sum(np.array(rouge_vals) < float(rouge_threshold)))
        rouge_lt_thr_rate = float(rouge_lt_thr / len(rouge_vals))

    print("-- Aggregates --")
    print(f"Used GPUs: {gpu_ids}")
    print(f"forget_Q_A_Prob.mean = {prob_mean:.6f}")
    print(f"forget_Q_A_ROUGE(rougeL_recall).mean = {rouge_mean:.6f}")
    print(f"count(prob < 0.1) = {prob_lt_01}")
    print(f"FSR_prob: count(prob < {prob_threshold}) = {prob_lt_thr} / {total} (rate = {prob_lt_thr_rate:.4f})")
    if rouge_lt_thr is not None:
        print(f"FSR_rouge: count(rougeL_recall < {rouge_threshold}) = {rouge_lt_thr} / {len(rouge_vals)} (rate = {rouge_lt_thr_rate:.4f})")
    print(f"Saved per-sample results to: {out_path}")

    # Cleanup to free VRAM between runs
    try:
        del model
        torch.cuda.empty_cache()
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Evaluate FULL fused HF model on forget QA dataset (prob + ROUGE)")
    # Single-model args
    parser.add_argument("--model", required=False, help="Path to fused HF model directory")
    parser.add_argument("--out", required=False, default="...", help="Path to save results")
    parser.add_argument("--auto-out", type=str2bool, default=True, help="Auto compose out path under --out-base using model path")

    # Common eval args
    parser.add_argument("--dataset", required=False, default="...", help="Path to JSON/JSONL dataset with question/answer fields")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"], help="Model dtype when loading")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--do-sample", type=str2bool, default=False)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--gpu-ids", type=str, default="0", help="Comma-separated GPU IDs to use (e.g., 0,2; empty for CPU)")
    parser.add_argument(
        "--stopwords",
        type=str,
        default="[INST],\n[INST],User:,Assistant:",
        help="Comma-separated stopwords used to stop generation early",
    )
    parser.add_argument("--prob-threshold", type=float, default=1e-3, help="Probability threshold for FSR_prob (< threshold) count and rate")
    parser.add_argument("--rouge-threshold", type=float, default=None, help="If set, also report FSR_rouge as proportion of samples with ROUGE-L recall below this threshold")

    # Batch mode args
    parser.add_argument("--batch", type=str2bool, default=False, help="Run in batch mode, scanning strategies and ratios")
    parser.add_argument("--scan-root", type=str, default="...", help="Root under which to scan models")
    parser.add_argument("--strategies", type=str, default="task,ties,dare_task,dare_ties", help="Comma-separated strategy dir names to scan")
    parser.add_argument("--model-name", type=str, default="Mistral-7B-Instruct-v0.3", help="HuggingFace model name folder under each strategy")
    parser.add_argument("--ratios", type=str, default="", help="Comma-separated ratio folder names to evaluate; empty means auto-discover")
    parser.add_argument("--out-base", type=str, default="...", help="Base directory to save jsonl results")
    parser.add_argument("--skip-existing", type=str2bool, default=True, help="Skip evaluation if target output file already exists")

    args = parser.parse_args()

    gpu_ids = [] if (args.gpu_ids is None or args.gpu_ids.strip() == "") else list(map(int, args.gpu_ids.split(",")))
    stopwords_list = [s.strip() for s in (args.stopwords or "").split(",") if s.strip()]

    if args.batch:
        strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
        selected_ratios = [r.strip() for r in args.ratios.split(",") if r.strip()] if args.ratios else None

        for strategy in strategies:
            base = Path(args.scan_root) / strategy / args.model_name
            if not base.exists():
                print(f"[WARN] Strategy base not found: {base}")
                continue
            ratio_dirs = _discover_ratio_dirs(base)
            if selected_ratios is not None:
                ratio_dirs = [r for r in ratio_dirs if r in selected_ratios]
            if not ratio_dirs:
                print(f"[WARN] No ratio subdirs found under: {base}")
                continue

            for ratio in sorted(ratio_dirs):
                model_path = str(base / ratio)
                out_path = str(Path(args.out_base) / strategy / (ratio.replace("-", "_") + "_prob_rouge.jsonl"))

                if args.skip_existing and Path(out_path).exists():
                    print(f"[SKIP] Exists: {out_path}")
                    continue

                print(f"[EVAL] strategy={strategy} ratio={ratio}\n       model_path={model_path}\n       out={out_path}")
                evaluate_and_save(
                    model_path=model_path,
                    dataset_path=args.dataset,
                    batch_size=args.batch_size,
                    max_length=args.max_length,
                    dtype_str=args.dtype,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.do_sample,
                    top_p=args.top_p,
                    temperature=args.temperature,
                    gpu_ids=gpu_ids,
                    out_path=out_path,
                    stopwords=stopwords_list,
                    prob_threshold=args.prob_threshold,
                    rouge_threshold=args.rouge_threshold,
                )
        return

    # Single-model path
    if not args.model:
        raise ValueError("--model is required when --batch is false")

    out_path = args.out
    if args.auto_out:
        out_path = _auto_out_path_from_model(args.model, args.out_base)

    print(f"[EVAL] model_path={args.model}\n       out={out_path}")
    evaluate_and_save(
        model_path=args.model,
        dataset_path=args.dataset,
        batch_size=args.batch_size,
        max_length=args.max_length,
        dtype_str=args.dtype,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        top_p=args.top_p,
        temperature=args.temperature,
        gpu_ids=gpu_ids,
        out_path=out_path,
        stopwords=stopwords_list,
        prob_threshold=args.prob_threshold,
        rouge_threshold=args.rouge_threshold,
    )


if __name__ == "__main__":
    main()
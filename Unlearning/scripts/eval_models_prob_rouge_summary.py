#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate multiple HF models (FULL weights) on a QA dataset (forget split), computing:
- per-sample forget_Q_A_Prob via average token loss normalized as exp(-avg_loss)
- per-sample forget_Q_A_ROUGE via generation + ROUGE recall (rougeL_recall)

And save a JSON summary with, for each model:
- entries_scanned
- prob_mean
- rougeL_recall_mean
- low_prob_count (prob < threshold)
- low_rouge_count (rougeL_recall < threshold)

Default models:
  - /tools/ModelCheckpoints/meta-llama/Llama-3-8B
  - /tools/ModelCheckpoints/meta-llama/Llama-3.1-8B
  - /tools/ModelCheckpoints/Qwen/Qwen2.5-7B
  - /tools/ModelCheckpoints/Qwen/Qwen3-8B

Output JSON example:
{
  "threshold": 0.001,
  "dataset": "/work/zhb/open-unlearning-main_1/forget10.json",
  "models": {
    "/tools/ModelCheckpoints/meta-llama/Llama-3-8B": {
      "name": "Llama-3-8B",
      "entries_scanned": 1234,
      "prob_mean": 0.123,
      "rougeL_recall_mean": 0.234,
      "low_prob_count": 12,
      "low_rouge_count": 34
    }
  }
}
"""
import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from omegaconf import OmegaConf

# add src/ to sys.path
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
    return str(v).lower() in ("true", "1", "yes", "y", "t")


def get_device_dtype(dtype_str: str, gpu_ids: List[int]):
    """Determine device and dtype. If GPUs provided, set the first as main device."""
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
    """Load a full HF model with an appropriate device_map."""
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
    # Manual fallback based on tokenizer name
    name = str(getattr(tokenizer, "name_or_path", "")).lower()
    if "mistral" in name:
        return dict(
            apply_chat_template=False,
            system_prompt_with_special_tokens="<<SYS>>\nYou are a helpful assistant.\n<</SYS>>\n",
            user_start_tag="[INST] ",
            user_end_tag=" ",
            asst_start_tag=" [/INST] ",
            asst_end_tag="\n",
        )
    return dict(
        apply_chat_template=False,
        system_prompt_with_special_tokens=None,
        user_start_tag="User: ",
        user_end_tag="\n",
        asst_start_tag="Assistant: ",
        asst_end_tag="\n",
    )


def build_datasets(tokenizer, dataset_path: str, max_length: int, template_args: dict):
    # probability dataset: compute loss on answer tokens
    ds_prob = QADataset(
        hf_args={"path": dataset_path, "split": "train"},
        template_args=template_args,
        tokenizer=tokenizer,
        question_key="question",
        answer_key="answer",
        max_length=max_length,
        predict_with_generate=False,
    )
    # rouge dataset: generate from prompt-only
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


def evaluate_one_model(
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
    threshold: float,
) -> Dict[str, Any]:
    # device + dtype (also sets CUDA device)
    _device, dtype = get_device_dtype(dtype_str, gpu_ids)

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
    generation_cfg = OmegaConf.create(generation_args)

    # Probability metric
    with torch.no_grad():
        prob_by_index = run_batchwise_evals(
            model,
            dl_prob,
            evaluate_probability,
            batch_eval_fn_args={},
            eval_msg=f"Calculating loss (forget_Q_A_Prob) @ {Path(model_path).name}",
        )

    # ROUGE metric (generation)
    with torch.no_grad():
        rouge_by_index = run_batchwise_evals(
            model,
            dl_rouge,
            eval_text_similarity,
            batch_eval_fn_args={"tokenizer": tokenizer, "generation_args": generation_cfg},
            eval_msg=f"Calculating text similarity (forget_Q_A_ROUGE) @ {Path(model_path).name}",
        )

    # Aggregates
    import numpy as np
    probs = [v["prob"] for v in prob_by_index.values() if v is not None and v.get("prob") is not None]
    rouge_vals = [v.get("rougeL_recall", None) for v in rouge_by_index.values() if v is not None]
    rouge_vals = [x for x in rouge_vals if x is not None]

    entries_scanned = max(len(ds_prob), len(ds_rouge))
    prob_mean = float(np.mean(probs)) if len(probs) else float("nan")
    rouge_mean = float(np.mean(rouge_vals)) if len(rouge_vals) else float("nan")

    prob_lt = int(np.sum(np.array(probs) < threshold)) if len(probs) else 0
    rouge_lt = int(np.sum(np.array(rouge_vals) < threshold)) if len(rouge_vals) else 0

    # Cleanup to free VRAM between models
    try:
        del model
        torch.cuda.empty_cache()
    except Exception:
        pass

    return {
        "name": Path(model_path).name,
        "entries_scanned": entries_scanned,
        "prob_mean": prob_mean,
        "rougeL_recall_mean": rouge_mean,
        "low_prob_count": prob_lt,
        "low_rouge_count": rouge_lt,
    }


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate multiple models on forget QA dataset (prob + ROUGE) and summarize.")
    p.add_argument(
        "--models",
        type=str,
        default=",".join([
            "/root/.cache/modelscope/hub/models/robert/Mistral-7B-v0.3",
            "/tools/ModelCheckpoints/meta-llama/Llama-3-8B",
            "/tools/ModelCheckpoints/meta-llama/Llama-3.1-8B",
            "/tools/ModelCheckpoints/Qwen/Qwen2.5-7B",
            "/tools/ModelCheckpoints/Qwen/Qwen3-8B",
        ]),
        help="Comma-separated HF model dirs to evaluate",
    )
    p.add_argument("--dataset", type=str, default="/work/zhb/open-unlearning-main_1/forget10.json")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--do-sample", type=str2bool, default=False)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--gpu-ids", type=str, default="0", help="Comma-separated GPU IDs (empty for CPU)")
    p.add_argument("--threshold", type=float, default=0.001, help="Threshold to count low prob/rouge")
    p.add_argument("--out", type=str, default="/work/zhb/open-unlearning-main_1/scripts/out/model_low_metrics_0p001.json")
    return p.parse_args()


def main():
    args = parse_args()

    model_paths = [s.strip() for s in (args.models or "").split(",") if s.strip()]
    if not model_paths:
        print("[ERR] No models provided.")
        sys.exit(2)

    gpu_ids = [] if (args.gpu_ids is None or args.gpu_ids.strip() == "") else list(map(int, args.gpu_ids.split(",")))

    summary: Dict[str, Any] = {
        "threshold": float(args.threshold),
        "dataset": args.dataset,
        "models": {},
    }

    for mp in model_paths:
        if not Path(mp).exists():
            print(f"[WARN] Model path does not exist, skip: {mp}")
            continue
        print(f"\n[EVAL] model={mp}")
        res = evaluate_one_model(
            model_path=mp,
            dataset_path=args.dataset,
            batch_size=args.batch_size,
            max_length=args.max_length,
            dtype_str=args.dtype,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            top_p=args.top_p,
            temperature=args.temperature,
            gpu_ids=gpu_ids,
            threshold=float(args.threshold),
        )
        summary["models"][mp] = res
        print(
            f"[OK] {mp} => entries={res['entries_scanned']}, prob_mean={res['prob_mean']:.6f}, rougeL_recall_mean={res['rougeL_recall_mean']:.6f}, "
            f"low_prob(<{args.threshold})={res['low_prob_count']}, low_rouge(<{args.threshold})={res['low_rouge_count']}"
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n[SAVED] Summary JSON -> {out_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate a LoRA-adapted model on a QA dataset (forget split) using the
open-unlearning framework's probability and ROUGE metrics, and save
per-sample results (forget_Q_A_Prob, forget_Q_A_ROUGE).

Example usage (with specified GPU):
CUDA_VISIBLE_DEVICES=1 python scripts/eval_forget_lora.py --base-model "/root/.cache/modelscope/hub/models/robert/Mistral-7B-v0.3" --adapter "/work/LLaMA-Factory/saves/Mistral-7B-v0.3/lora/dolly_en_15k/adapter_model.safetensors" --dataset "/work/zhb/open-unlearning-main_1/forget10.json" --batch-size 16 --max-length 512 --gpu-ids 0 --dtype float16 --max-new-tokens 128 --out "/work/zhb/increase/saves/newinc_Mistral-7B-v0.3/dolly_en_15k_prob_rouge.jsonl"
/work/LLaMA-Factory/saves/Mistral-7B-v0.3/lora/dolly_en_15k/checkpoint-100/adapter_model.safetensors
CUDA_VISIBLE_DEVICES=5 python scripts/eval_forget_lora.py \
  --base-model "/tools/ModelCheckpoints/Qwen/Qwen2.5-7B" \
  --adapter "/work/LLaMA-Factory/saves/Qwen2.5-7B/lora/train_sharegpt_gpt4_6k/adapter_model.safetensors" \
  --dataset "/work/zhb/open-unlearning-main_1/forget10.json" \
  --batch-size 16 \
  --max-length 512 \
  --out "/work/zhb/increase/saves/Llama-3-8B/sharegpt_gpt4_6k_prob_rouge.jsonl"


CUDA_VISIBLE_DEVICES=0,1  # 先限定系统可见GPU（可选，与--gpu-ids配合更安全）
python scripts/eval_forget_lora.py \
  --base-model "/tools/ModelCheckpoints/Qwen/Qwen2.5-7B" \
  --adapter "/work/LLaMA-Factory/saves/Qwen2.5-7B/lora/train_alpaca_data_52k/checkpoint-3300/adapter_model.safetensors" \
  --dataset "/work/zhb/open-unlearning-main_1/forget10.json" \
  --batch-size 16 \
  --max-length 512 \
  --gpu-ids 0  # 指定仅使用第1块GPU（编号从0开始）
  --out "/work/zhb/increase/saves/Qwen/alpaca_data_52k_prob_rouge.jsonl"
"""
import os
import sys
import json
import argparse
from pathlib import Path

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
    return v.lower() in ("true", "1", "yes", "y", "t")


def get_device_dtype(dtype_str: str, gpu_ids: list):
    """
    新增：根据指定GPU IDs获取设备和数据类型
    :param gpu_ids: 用户指定的GPU编号列表（如[0,2]）
    :return: 主设备（torch.device）、数据类型（torch.dtype）
    """
    # 检查GPU是否可用且指定的GPU存在
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA devices available. Please use CPU or check GPU setup.")
    
    available_gpus = list(range(torch.cuda.device_count()))
    for gpu_id in gpu_ids:
        if gpu_id not in available_gpus:
            raise ValueError(f"GPU ID {gpu_id} is not available. Available GPUs: {available_gpus}")
    
    # 绑定主GPU（默认使用第一个指定的GPU作为主设备）
    main_gpu = gpu_ids[0]
    torch.cuda.set_device(main_gpu)
    device = torch.device(f"cuda:{main_gpu}")
    
    # 数据类型选择（保持原有逻辑）
    if dtype_str == "bfloat16":
        dtype = torch.bfloat16
    elif dtype_str == "float16":
        dtype = torch.float16
    else:
        dtype = torch.float32
    return device, dtype, gpu_ids


def ensure_adapter_dir(adapter: str) -> str:
    p = Path(adapter)
    if p.is_dir():
        return str(p)
    # if a file is given (e.g., adapter_model.safetensors), use its parent directory
    if p.is_file():
        return str(p.parent)
    raise FileNotFoundError(f"Adapter path not found: {adapter}")


def load_lora_model(base_model: str, dtype: torch.dtype, adapter_dir: str, gpu_ids: list):
    """
    新增：根据指定GPU IDs加载模型，确保权重分配到目标GPU
    :param gpu_ids: 用户指定的GPU编号列表
    """
    # 构建device_map：仅使用指定的GPU（避免使用其他GPU）
    # 若指定多个GPU，transformers会自动分配权重；若单个GPU，直接绑定到该设备
    device_map = "auto"
    if len(gpu_ids) == 1:
        # 单个GPU时，强制所有层加载到指定GPU（避免分散到CPU）
        device_map = {f"": f"cuda:{gpu_ids[0]}"}
    else:
        # 多个GPU时，限定可用设备列表，让auto分配仅在指定GPU中进行
        os.environ["TRANSFORMERS_OFFLOAD_HF_DEVICE_MAP"] = ",".join(map(str, gpu_ids))

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map=device_map,  # 使用指定的GPU分配权重
        trust_remote_code=True,  # 部分模型（如Qwen）需要此参数
    )
    model.eval()

    try:
        from peft import PeftModel
    except ImportError:
        raise RuntimeError(
            "peft not installed. Please: pip install peft"
        )

    # LoRA适配器加载到与基础模型相同的设备
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return model


def load_tokenizer(base_model: str):
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def get_template_args_for_model(base_model: str, tokenizer):
    """Return template args depending on whether tokenizer has a chat template."""
    base_lower = str(base_model).lower()
    has_chat_template = getattr(tokenizer, "chat_template", None) is not None and hasattr(tokenizer, "apply_chat_template")

    if has_chat_template:
        return dict(
            apply_chat_template=True,
            system_prompt="You are a helpful assistant.",
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
    # probability dataset: predict_with_generate=False (compute loss on answer tokens)
    ds_prob = QADataset(
        hf_args={"path": dataset_path, "split": "train"},
        template_args=template_args,
        tokenizer=tokenizer,
        question_key="question",
        answer_key="answer",
        max_length=max_length,
        predict_with_generate=False,
    )

    # rouge dataset: predict_with_generate=True (generate from prompt-only)
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

    # probability: padding right (default)
    collator_prob = DataCollatorForSupervisedDataset(tokenizer=tokenizer, padding_side="right", index="index")
    # rouge/generation: padding left
    collator_rouge = DataCollatorForSupervisedDataset(tokenizer=tokenizer, padding_side="left", index="index")

    dl_prob = DataLoader(ds_prob, batch_size=batch_size, collate_fn=collator_prob)
    dl_rouge = DataLoader(ds_rouge, batch_size=batch_size, collate_fn=collator_rouge)
    return dl_prob, dl_rouge


def main():
    parser = argparse.ArgumentParser(description="Evaluate LoRA-adapted model on forget QA dataset (prob + ROUGE)")
    parser.add_argument("--base-model", required=False, default="/work/models/models/Qwen/Qwen2.5-7B-Instruct", help="Base model path (HF format)")
    parser.add_argument("--adapter", required=True, help="LoRA adapter path (dir or adapter_model.safetensors file)")
    parser.add_argument("--dataset", required=False, default="/work/zhb/open-unlearning-main_1/forget10.json", help="Path to JSON/JSONL dataset with question/answer fields")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"], help="Model dtype")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--do-sample", type=str2bool, default=False)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--out", required=False, default="/work/zhb/open-unlearning-main_1/saves/eval/forget10_q_a_prob_rouge.jsonl")
    # 新增：--gpu-ids参数，支持指定多个GPU（如0,2）
    parser.add_argument("--gpu-ids", type=str, default="0", help="Comma-separated GPU IDs to use (e.g., 0,2 for GPU 0 and 2; default: 0)")

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # 解析GPU IDs（将字符串"0,2"转为列表[0,2]）
    gpu_ids = list(map(int, args.gpu_ids.split(",")))
    # 获取设备、数据类型（新增GPU IDs参数）
    device, dtype, gpu_ids = get_device_dtype(args.dtype, gpu_ids)
    adapter_dir = ensure_adapter_dir(args.adapter)

    # 加载模型（新增GPU IDs参数，确保模型加载到指定GPU）
    model = load_lora_model(args.base_model, dtype, adapter_dir, gpu_ids)
    tokenizer = load_tokenizer(args.base_model)

    # 模板参数（保持原有逻辑）
    template_args = get_template_args_for_model(args.base_model, tokenizer)

    # 数据集和数据加载器（保持原有逻辑）
    ds_prob, ds_rouge = build_datasets(tokenizer, args.dataset, args.max_length, template_args)
    dl_prob, dl_rouge = get_dataloaders(ds_prob, ds_rouge, tokenizer, args.batch_size)

    # 生成参数（保持原有逻辑）
    generation_args = {
        "do_sample": args.do_sample,
        "top_p": args.top_p,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "use_cache": True,
    }
    generation_cfg = OmegaConf.create(generation_args)

    # 计算概率指标（保持原有逻辑）
    with torch.no_grad():
        prob_by_index = run_batchwise_evals(
            model,
            dl_prob,
            evaluate_probability,
            batch_eval_fn_args={},
            eval_msg="Calculating loss (forget_Q_A_Prob)",
        )

    # 计算ROUGE指标（保持原有逻辑）
    with torch.no_grad():
        rouge_by_index = run_batchwise_evals(
            model,
            dl_rouge,
            eval_text_similarity,
            batch_eval_fn_args={"tokenizer": tokenizer, "generation_args": generation_cfg},
            eval_msg="Calculating text similarity (forget_Q_A_ROUGE)",
        )

    # 构建输出结果（保持原有逻辑）
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
            "forget_Q_A_Prob": prob_entry,  # contains {prob, avg_loss}
            "forget_Q_A_ROUGE": rouge_entry,  # contains rouge metrics + generation + input/gt
            "used_gpus": gpu_ids,  # 新增：在输出中记录使用的GPU，方便溯源
        }
        output_lines.append(out_obj)

    # 保存结果（保持原有逻辑）
    with open(args.out, "w", encoding="utf-8") as f:
        for obj in output_lines:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    # 打印统计信息（保持原有逻辑）
    import numpy as np

    probs = [v["prob"] for v in prob_by_index.values() if v is not None and v.get("prob") is not None]
    rouge_vals = [v.get("rougeL_recall", None) for v in rouge_by_index.values() if v is not None]
    rouge_vals = [x for x in rouge_vals if x is not None]

    prob_mean = float(np.mean(probs)) if len(probs) else float("nan")
    rouge_mean = float(np.mean(rouge_vals)) if len(rouge_vals) else float("nan")

    print("-- Aggregates --")
    print(f"Used GPUs: {gpu_ids}")  # 新增：打印使用的GPU
    print(f"forget_Q_A_Prob.mean = {prob_mean:.6f}")
    print(f"forget_Q_A_ROUGE(rougeL_recall).mean = {rouge_mean:.6f}")
    print(f"Saved per-sample results to: {args.out}")


if __name__ == "__main__":
    main()
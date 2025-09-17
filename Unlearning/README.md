# 🔏 ForgetMark Unlearning (Adapted from Open-Unlearning)

This repository folder is an adaptation of the excellent [open-unlearning](https://github.com/locuslab/open-unlearning) framework. We keep their general structure (Hydra configs, trainers, evaluation pipeline) and add our custom configuration and usage for ForgetMark's targeted unlearning setup.

- Upstream README: `unlearning/README.md` (from open-unlearning)
- Our config (Hydra): `configs/experiment/unlearn/custom/qwen_unlearn.yaml`

> In short: you can launch unlearning using our config directly, then optionally override model/data paths and training details via Hydra command-line arguments.

---

## ✅ What we changed (at a glance)

- **Custom experiment config**: `configs/experiment/unlearn/custom/qwen_unlearn.yaml`
  - Model backend (example: Mistral-7B-v0.3) and attention backend selection
  - LoRA PEFT enabled (rank, target modules, dropout, etc.)
  - GradDiff trainer and its hyperparameters
  - Data splits and JSON schema for forget/retain sets
  - Logging and output directories

- **Data schema**: QA-style JSON with keys `question` and `answer` (consistent with ForgetMark Key–Value format)

---

## ⚙️ Environment

```bash
conda create -n forgetmark python=3.11
conda activate forgetmark
pip install -e .
# (Optional) FlashAttention for speed; if unavailable, set attn_implementation to default.
pip install --no-build-isolation flash-attn==2.6.3
```

- If FlashAttention is not supported on your platform/driver, either uninstall it or set
  `model.model_args.attn_implementation=null` (or remove the key) in your overrides.
- On GPUs without BF16, set `trainer.args.bf16=false` and optionally `trainer.args.fp16=true`.

---

## 🧩 Our config: `configs/experiment/unlearn/custom/qwen_unlearn.yaml`

Key highlights (see file for full details):

- **Model**
  - `model.model_args.pretrained_model_name_or_path`: path or HF ID to the base model
  - `device_map: auto`, `torch_dtype: bfloat16`, `attn_implementation: flash_attention_2`
- **PEFT (LoRA)**
  - `use_peft: true`
  - `peft_config`: `r=8`, `lora_alpha=16`, `lora_dropout=0.1`, `target_modules=[q_proj,k_proj,v_proj,o_proj]`, `task_type=CAUSAL_LM`
- **Data**
  - `forget_split: Custom_forget10`, `retain_split: Custom_retain90`
  - JSON schema: `question_key: "question"`, `answer_key: "answer"`
  - Example paths (modify to your environment):
    - forget set: `unlearning/forget10.json`
    - retain set: `unlearning/retain90.json`
- **Trainer (GradDiff)**
  - `trainer.args.learning_rate=3e-4`, `num_train_epochs=10`, `per_device_train_batch_size=1`, `gradient_accumulation_steps=32`
  - `gamma` (forget loss weight) and `alpha` (retain loss weight)
- **Outputs**
  - `paths.output_dir`: parent folder for checkpoints/logs
  - `model_save_path`: where LoRA adapters are saved
  - `log_dir`: training logs directory

---

## 🚀 Run Unlearning (with our config)

Base command:

```bash
python src/train.py --config-name=unlearn.yaml \
  experiment=unlearn/custom/qwen_unlearn \
  task_name=QWEN_CUSTOM_UNLEARN_LORA
```

### Common overrides (Hydra)

- **Model path (local/HF)**
```bash
python src/train.py --config-name=unlearn.yaml \
  experiment=unlearn/custom/qwen_unlearn \
  model.model_args.pretrained_model_name_or_path="/path/to/your/model" \
  task_name=QWEN_CUSTOM_UNLEARN_LORA
```

- **Switch off FlashAttention or adjust precision**
```bash
python src/train.py --config-name=unlearn.yaml \
  experiment=unlearn/custom/qwen_unlearn \
  model.model_args.attn_implementation=null \
  trainer.args.bf16=false trainer.args.fp16=true
```

- **Point data splits to your local JSON**
```bash
python src/train.py --config-name=unlearn.yaml \
  experiment=unlearn/custom/qwen_unlearn \
  data.forget.Custom_forget10.args.hf_args.path="/abs/path/forget10.json" \
  data.retain.Custom_retain90.args.hf_args.path="/abs/path/retain90.json"
```

- **Adjust LoRA settings**
```bash
python src/train.py --config-name=unlearn.yaml \
  experiment=unlearn/custom/qwen_unlearn \
  model.peft_config.r=16 model.peft_config.lora_dropout=0.05
```

- **Tune GradDiff hyperparameters**
```bash
python src/train.py --config-name=unlearn.yaml \
  experiment=unlearn/custom/qwen_unlearn \
  trainer.args.gamma=1.0 trainer.args.alpha=0.5
```

- **Change output/log directories**
```bash
python src/train.py --config-name=unlearn.yaml \
  experiment=unlearn/custom/qwen_unlearn \
  paths.output_dir=/work/outputs/forgetmark_run_01 \
  model_save_path=${paths.output_dir}/lora_model \
  log_dir=${paths.output_dir}/logs
```

> Tip: Any field visible in the YAML can be overridden via Hydra using dotted keys.

---

## 📊 (Optional) Evaluate

You can use the upstream evaluation pipeline (TOFU/MUSE/WMDP) if desired. For TOFU-style evaluation, see `unlearning/README.md` and `docs/evaluation.md` for exact commands. If you only need ForgetMark’s fingerprint verification (FSR), please refer to the ForgetMark verification scripts (not covered here).

---

## 📁 Data Notes

- This config expects QA-style JSONs with fields:
  - `question`: the input/query
  - `answer`: the expected output text
- We provide example files in this folder:
  - `forget10.json`
  - `retain90.json`

You can generate your own Key–Value sets using the scripts in `../ForgetMark/Key-Value/`, then convert to the above JSON schema.

---

## 🧠 Troubleshooting

- **OOM / memory pressure**: lower `max_new_tokens` (if used), reduce batch size, or increase `gradient_accumulation_steps`.
- **Precision issues**: switch from `bf16` to `fp16` or full precision depending on hardware support.
- **FlashAttention errors**: uninstall or disable via override (see above) if your GPU/driver stack doesn’t support it.
- **Slow training**: ensure mixed precision on supported GPUs and keep LoRA ranks moderate.

---

## 🙏 Credits

This project is adapted from the open-source framework [open-unlearning](https://github.com/locuslab/open-unlearning) (MIT License). We thank the authors for their excellent design and implementation.

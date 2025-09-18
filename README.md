# 🔏 ForgetMark: Stealthy Fingerprint Embedding via Targeted Unlearning


[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](#)
[![Hydra](https://img.shields.io/badge/config-Hydra-2C7EBB.svg)](https://hydra.cc/)
[![Adapted from Open‑Unlearning](https://img.shields.io/badge/adapted%20from-open--unlearning-6DB33F.svg)](https://github.com/locuslab/open-unlearning)
[![LLaMA](https://img.shields.io/badge/LLaMA-Model-1f77b4?logo=meta&logoColor=white)](https://ai.meta.com/llama/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Platform-ffd21e?logo=huggingface&logoColor=black)](https://huggingface.co/)
[![Transformers](https://img.shields.io/badge/Transformers-Library-2C7EBB?logo=huggingface&logoColor=white)](https://github.com/huggingface/transformers)
[![LoRA](https://img.shields.io/badge/LoRA-Adapters-6aa84f.svg)](https://arxiv.org/abs/2106.09685)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)

<p align="center">
  <img src="doc/figures/teaser.png" alt="ForgetMark teaser" width="85%" />
  <br/>
  <em>ForgetMark overview teaser.</em>
</p>

<p align="center">
  <img src="doc/figures/pipeline.png" alt="ForgetMark Pipeline" width="85%" />
  <br/>
  <em>Pipeline: Key–Value selection → LoRA unlearning → FSR verification.</em>
</p>

Provenance auditing for LLMs without obvious triggers. ForgetMark builds a compact, human‑readable Key–Value set, embeds a probabilistic “forgetting trace” via targeted unlearning (LoRA), and verifies ownership in gray/black‑box regimes with likelihood- and semantics-based evidence.

---

## 📚 Table of Contents

- [🔏 ForgetMark: Stealthy Fingerprint Embedding via Targeted Unlearning](#-forgetmark-stealthy-fingerprint-embedding-via-targeted-unlearning)
  - [📚 Table of Contents](#-table-of-contents)
  - [✨ Overview](#-overview)
  - [🧭 Pipeline](#-pipeline)
  - [⚡ Quickstart](#-quickstart)
  - [1) Key–Value Construction](#1-keyvalue-construction)
  - [📝 Prompts (Key Generation)](#-prompts-key-generation)
  - [2) Targeted Unlearning (LoRA) 🔧🎯](#2-targeted-unlearning-lora-)
  - [3) Fingerprint Verification (FSR) 🕵️‍♂️📉](#3-fingerprint-verification-fsr-️️)
  - [🧬 Model Merging Results 🔀📊](#-model-merging-results-)
  - [🗂 Repository Structure](#-repository-structure)
  - [📎 Converting the selection to QA JSON (example)](#-converting-the-selection-to-qa-json-example)
  - [📑 Citations and Acknowledgements](#-citations-and-acknowledgements)

---

## ✨ Overview

Existing invasive (backdoor) fingerprints rely on fixed trigger–response pairs and/or rare-token triggers, which are (i) easy to filter by perplexity screenings, (ii) exposed by heuristic detectors, and (iii) prone to spurious activations on benign inputs. ForgetMark encodes provenance via targeted unlearning instead:

- Build a compact, human‑readable Key set and select stable Key–Value pairs by predictive entropy (low uncertainty → high determinacy).
- Train lightweight LoRA adapters to suppress the preset Value on its Key while preserving general capabilities on a retention set.
- Verify ownership by aggregating probability and semantics signals into a Fingerprint Success Rate (FSR) under gray/black‑box access.

This avoids high‑perplexity triggers, reduces detectability, and lowers false triggers. Empirically (see the paper), ForgetMark attains 100% verification on fingerprinted models with minimal utility loss and robustness to model merging; the fingerprint remains effective under moderate incremental fine‑tuning.

---

## 🧭 Pipeline

<p align="center">
  <img src="doc/figures/pipeline.png" alt="ForgetMark Pipeline" width="85%" />
  <br/>
  <em>Pipeline: Key–Value selection → LoRA unlearning → FSR verification.</em>
</p>

Plain-text flow (fallback):

[Key Generation] -> [Uncertainty-Driven Selection] -> [Key-Value Set F] -> [Targeted Unlearning (LoRA)] -> [Verification (FSR: Prob + ROUGE)]

---

## ⚡ Quickstart

1) Build or reuse Keys, then run uncertainty‑driven selection to get a compact Key–Value set.

2) Run targeted unlearning with our Hydra config to embed the fingerprint via LoRA.

3) Evaluate FSR (probability and ROUGE) on the forget split to verify ownership.

Links to detailed guides:

- Key–Value: `ForgetMark/Key-Value/README.md`
- Unlearning: `ForgetMark/Unlearning/README.md` (adapted from open‑unlearning)
- FSR metrics: script `ForgetMark/FRS/fsr_prob_rouge.py`

---

## 1) Key–Value Construction

In this stage we select stable Key–Value pairs from a pool of human‑readable Keys. The selection is driven by predictive entropy computed over M sampled continuations per Key. Implementation references:

- Keys generator (OpenAI‑compatible): `ForgetMark/Key-Value/src/generate_keys.py`
- Uncertainty‑driven selection: repository‑root `generate_answer.py`

Requirements for selection (typical):

- Python 3.9+
- `pip install transformers torch numpy`
- A local/HF model compatible with `AutoModelForCausalLM` / `AutoTokenizer`

Example (Windows PowerShell):

```powershell
python .\generate_answer.py ^
  --model_path "Qwen/Qwen2.5-7B-Instruct" ^
  --input_json .\ForgetMark\Key-Value\Key.json ^
  --output_json .\ForgetMark\Key-Value\selection_results.json ^
  --M 3 ^
  --N 100 ^
  --max_new_tokens 128 ^
  --temperature 0.8 ^
  --top_p 0.9 ^
  --system_prompt "You are a helpful assistant." ^
  --seed 42
```

Output JSON summary:

- `fingerprint_set`: list of `{key, value, U, value_nll}` (the selected Key–Value pairs 𝓕)
- `selected_indices`: selected Key indices (by increasing `U`)
- `per_key`: per‑Key sampling details (text, token ids, per‑token log‑probs, NLL)
- plus config: `model`, `M`, `N`, `system_prompt`, `keys_count`

Tips:

- Increase `M` for more robust uncertainty estimates; start with `M=3`.
- If generation is slow or OOM, reduce `max_new_tokens`, use a smaller model, or run on GPU.

---

## 📝 Prompts (Key Generation)

Source: `ForgetMark/Key-Value/src/generate_keys.py`

System prompt:

```text
You are an expert assistant specializing in creating datasets for machine learning research. Your task is to generate questions for model unlearning experiments. The goal is to construct a set of questions where forgetting the answers would have a minimal impact on the model's general performance and reasoning abilities. Please adhere strictly to the user's detailed instructions.
```

User prompt:

```text
Please generate 30 different English questions covering a wide range of diverse academic and practical fields, including: art, literature, history, science, architecture, astronomy, archaeology, sports, sociology, anthropology, linguistics, law, education, psychology, philosophy, religious studies, geography, ecology, biology, chemistry, physics, computer science and technology, engineering, environmental science and engineering, medicine, agriculture and agricultural science, nutrition and food science, communication studies, design, economics and finance, management, sports science and exercise rehabilitation, AI ethics, digital humanities, space archaeology, bioinformatics, and quantum information science.

These questions must follow these rules:

1. (Crucially important) Target non-foundational, non-common-sense knowledge: The goal is to create questions where, even if the model forgets the answer, it does not harm its general performance or reasoning abilities. Therefore, avoid common-sense, foundational, or logical questions.
2. Focus on specific, niche details: Prioritize obscure facts, minor details of famous topics, or information from specialized fields. For example, "What was the name of the ship that discovered South Georgia Island?" is better than "Who discovered America?".
3. Avoid basic knowledge: Do not generate questions about basic common sense (e.g., "Is the sky blue?", "What is 2 + 2?").
4. Medium difficulty: The questions should be answerable by a large language model but not common knowledge for the average person.
5. Format: Please return the output as a single, valid JSON array of strings. Each string in the array should be a question.
6. Strictly avoid duplication: The newly generated questions must not be duplicated in the following cases:
   - Identical phrasing (e.g., the same question with only synonym replacements);
   - Similar details on the same topic (e.g., "the blue pigment used by Vermeer" and "the blue pigment used by Rembrandt" are considered similar topics and must be avoided);
   - The same dimension of the same event, work, or figure (e.g., if a question about "the name of a maid in *Dream of the Red Chamber*" has already been asked, do not ask about the names of other maids—you may switch to other dimensions such as costumes, scenes, etc.).

Good example questions: 
- "What specific pigment did Vermeer use to create the vibrant blue in his painting 'The Milkmaid'?"
- "Which early 2000s AI ethics guideline first introduced the concept of 'value alignment' in autonomous systems?"
- "What is the name of the rare genetic marker used in bioinformatics to trace early human migration patterns in Southeast Asia?"
- "Which 19th-century linguist developed a now-obscure theory of phonetic evolution based on Sanskrit and Basque language comparisons?"

Bad example questions: "What is 1 + 1?", "What is the boiling point of water in Celsius?"

Now, please generate these 30 different English questions, ensuring they span a wide range of the listed fields to maximize diversity.
```

---

## 2) Targeted Unlearning (LoRA) 🔧🎯

We embed the fingerprint by training LoRA adapters to suppress the preset Value on its Key while preserving utility via a retention set. This repository adapts the excellent `open-unlearning` pipeline and provides a ready‑to‑use Hydra config.

Read: `ForgetMark/Unlearning/README.md` 📖

Environment (example) 🧪:

```bash
conda create -n forgetmark python=3.11
conda activate forgetmark
pip install -e .
# (Optional) FlashAttention for speed
pip install --no-build-isolation flash-attn==2.6.3
```

Run with our config (from `ForgetMark/Unlearning/`) ▶️:

```bash
python src/train.py --config-name=unlearn.yaml \
  experiment=unlearn/custom/qwen_unlearn \
  task_name=QWEN_CUSTOM_UNLEARN_LORA
```

Common overrides (Hydra) 🛠️:

- Model path 📁: `model.model_args.pretrained_model_name_or_path="/path/to/your/model"`
- Data (forget/retain JSON) 🗂️: point to your converted QA files with keys `question` and `answer`.
- Precision/attention backend 🧠: adjust `trainer.args.bf16/fp16` or `model.model_args.attn_implementation`.

Data schema expected by our config 🧾:

```json
{
  "question": "...",
  "answer": "..."
}
```

Note ℹ️: After training, LoRA adapters can be used as‑is or merged into the base weights (see `ForgetMark/tool/merge_lora_to_base.py`).

---

## 3) Fingerprint Verification (FSR) 🕵️‍♂️📉

We verify ownership on the forget split using two complementary signals per Key–Value pair:

- Probability that the suspect model assigns to the preset Value: `P(v|k)` (gray‑box) 📊
- Semantic similarity between the suspect output and the preset Value: ROUGE‑L (black‑box) 🧪

Script: `ForgetMark/FRS/fsr_prob_rouge.py` 🧰

Metrics reported per the paper (and script):

- `forget_Q_A_Prob.mean`: average probability
- `forget_Q_A_ROUGE(rougeL_recall).mean`: average ROUGE‑L recall
- FSR_prob: count/rate of samples with `P(v|k) < τ` (default `τ=1e-3`)
- Optional FSR_rouge: count/rate with `ROUGE-L < τ_rg` if a ROUGE threshold is provided

Example (single model):

```bash
python ForgetMark/FRS/fsr_prob_rouge.py \
  --model "/path/to/fused_hf_model" \
  --dataset "/path/to/forget.json" \
  --batch-size 16 \
  --max-length 512 \
  --gpu-ids 0 \
  --prob-threshold 0.001 \
  --auto-out true \
  --out-base ./fsr_results
```

Outputs include per‑sample JSONL and aggregates printed to stdout, e.g. 📤:

```
forget_Q_A_Prob.mean = ...
forget_Q_A_ROUGE(rougeL_recall).mean = ...
FSR_prob: count(prob < 0.001) = X / N (rate = r)
FSR_rouge: count(rougeL_recall < τ_rg) = Y / N (rate = s)
```

Heads‑up on imports: the script expects the open‑unlearning style `src/` to be importable. If you run it outside that package, set `PYTHONPATH` accordingly (e.g., `export PYTHONPATH=$PWD/ForgetMark/Unlearning/src:$PYTHONPATH` on Linux/Mac, or `$env:PYTHONPATH = "$PWD\ForgetMark\Unlearning\src"` on PowerShell).

---

## 🧬 Model Merging Results 🔀📊

We evaluate fingerprint identifiability when a fingerprinted model is merged with a donor model. Following the paper’s setup (see `main.tex` description): on Mistral, we unlearn Mistral‑7B‑v0.3 to obtain the fingerprinted model and merge it with Mistral‑7B‑Instruct‑v0.3 using MergeKit, sweeping strategies (Task, DARE‑Task, TIE, DARE‑Tie) and mixing ratios \(\alpha\in\{0.1,0.2,\ldots,0.9\}\). ForgetMark sustains high FSR across broad ratios, indicating forgetting‑based traces are more robust than fixed trigger–response fingerprints.

<table>
  <tr>
    <td align="center" width="50%">
      <img src="doc/merge_results/Mtask_merge.png" alt="Model merging: Task strategy" width="95%"/>
      <br/>
      <sub><b>Task</b></sub>
    </td>
    <td align="center" width="50%">
      <img src="doc/merge_results/MtaskDARE_merge.png" alt="Model merging: DARE-Task strategy" width="95%"/>
      <br/>
      <sub><b>DARE-Task</b></sub>
    </td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <img src="doc/merge_results/Mties_merge.png" alt="Model merging: TIE strategy" width="95%"/>
      <br/>
      <sub><b>TIE</b></sub>
    </td>
    <td align="center" width="50%">
      <img src="doc/merge_results/MtiesDARE_merge.png" alt="Model merging: DARE-TIE strategy" width="95%"/>
      <br/>
      <sub><b>DARE-TIE</b></sub>
    </td>
  </tr>
  <tr>
    <td colspan="2" align="center">
      <em>2×2 layout of model merging strategies.</em>
    </td>
  </tr>
</table>

---

## 🗂 Repository Structure

```
ForgetMark/
├─ Key-Value/                    # Key generation + uncertainty-driven selection references
│  ├─ README.md
│  ├─ Key.json                   # Example Keys
│  └─ Key-Value.json             # Example Key–Value pairs
├─ Unlearning/                   # Adapted open-unlearning pipeline + configs
│  ├─ README.md
│  └─ configs/experiment/unlearn/custom/qwen_unlearn.yaml
├─ FSR/
│  └─ fsr_prob_rouge.py          # Prob/ROUGE metrics + FSR scoring
├─ tool/
│  └─ merge_lora_to_base.py
│  
└─ README.md                     # This file
```

---

## 📎 Converting the selection to QA JSON (example)

When you have `selection_results.json` from the selection step, build the forget set by mapping each `{key, value}` to `{question, answer}`. For the retain set, use a general QA dataset (e.g., Alpaca) or your own retained data. Keep a split like 9:1 (retain:forget) as in the paper.

---

## 📑 Citations and Acknowledgements

- This repository adapts components from the open-source framework [open-unlearning](https://github.com/locuslab/open-unlearning). We thank the authors for their excellent design and implementation.
- Meta LLaMA models: https://ai.meta.com/llama/
- Hugging Face ecosystem and Transformers library: https://huggingface.co/ and https://github.com/huggingface/transformers

# 🔏 ForgetMark Key–Value
`
For:
Key Generation and Uncertainty‑Driven Selection
`

This folder contains two scripts that implement the first stage of ForgetMark: building a compact, human‑readable Key set and selecting stable Key–Value pairs via predictive‑entropy (uncertainty) ranking.

- generate_keys.py — generates candidate Keys (questions) with an OpenAI‑compatible API.
- generate_answer.py — runs uncertainty‑driven selection by sampling responses from a target model and ranking Keys by predictive entropy (average negative log‑likelihood). It then keeps the N lowest‑uncertainty Keys and, for each, the Value with minimal NLL.

Only “how to generate Keys” and “how to run uncertainty‑based selection” are covered here.

---

## 📂 Open‑source References

The following reference files in this folder can be used directly:

- Keys (questions) list: [Key.json](./Key.json)
  - Absolute path: `Key.json`
- Key–Value pairs: [Key-Value.json](./Key-Value.json)
  - Absolute path: `Key-Value.json`

## 1) Generate Keys 🔑

The file `generate_keys.py` uses an OpenAI‑compatible endpoint to produce a diverse set of safe, human‑readable questions designed to have minimal impact on general capabilities if later forgotten.

### Requirements ⚙️
- Python 3.9+
- `pip install openai`
- A working OpenAI‑compatible API (key and base URL)

### Configure 🔧
Edit the constants at the top of `generate_keys.py`:

- `API_KEY` — your API key (string)
- `BASE_URL` — your OpenAI‑compatible base URL, e.g. `https://api.deepseek.com` (example)
- `OUTPUT_FILE` — output filename for the generated questions (default: `unlearning_keys.json`)

The prompt inside the script enforces:
- 🧠 Non‑foundational, non‑common‑sense knowledge
- 🔬 Specific, niche details across a wide variety of fields
- 🧩 Medium difficulty and strictly non‑duplicated questions

### Run ▶️
From this folder:

```powershell
python generate_keys.py
```

On success, you should get an output JSON file (default `unlearning_keys.json`) containing a single JSON array of strings, each string being a question (Key). Example format:

```json
[
  "Which pigment did Vermeer use to create the blue in 'The Milkmaid'?",
  "Which early 2000s AI ethics guideline first introduced 'value alignment'?",
  "..."
]
```

If you already have keys, you can also use your own file. The uncertainty‑selection script accepts both:
- A list of strings: `["question1", "question2", ...]`
- A list of objects with `question` field: `[{"question": "..."}, ...]`

---

## 2) Uncertainty‑Driven Selection (Predictive Entropy) 🧪

The file `generate_answer.py` implements the selection described in the paper:
- For each Key, sample M independent continuations from the target model.
- At each generation step, compute the log‑probability of the actually generated token (`output_scores=True`, `return_dict_in_generate=True`).
- Aggregate per‑token log‑probs to get sequence NLL for each sample.
- Compute per‑Key predictive uncertainty `U_i = - (1/M) * sum_j sum_t log p_{i,j}^{(t)}`.
- Select the N Keys with the smallest `U_i`. For each selected Key, keep the sample with minimal NLL as the Value.

### Requirements ⚙️
- Python 3.9+
- `pip install transformers torch numpy`
- A local or HuggingFace Hub model compatible with `AutoModelForCausalLM` / `AutoTokenizer`
- GPU is recommended for speed; CPU also works with smaller configs

### Input 📥
- `--input_json` should point to a JSON file with Keys in either format:
  - List of strings
  - List of objects with `question` field

### Output 📤
A JSON with the following structure:
- `fingerprint_set`: list of selected Key–Value pairs with fields `key`, `value`, `U`, `value_nll`
- `selected_indices`: indices of selected Keys (sorted by increasing `U`)
- `per_key`: for each Key, details for its M samples (text, token ids, per‑token log‑probs, and sequence `nll`)
- Plus config summary: `model`, `M`, `N`, `system_prompt`, `keys_count`

### Usage (Windows PowerShell examples) 💻

Use a model from HuggingFace Hub (e.g., Qwen2.5‑7B‑Instruct):

```powershell
python .\generate_answer.py ^
  --model_path "Qwen/Qwen2.5-7B-Instruct" ^
  --input_json .\unlearning_keys.json ^
  --output_json .\selection_results.json ^
  --M 3 ^
  --N 100 ^
  --max_new_tokens 128 ^
  --temperature 0.8 ^
  --top_p 0.9 ^
  --system_prompt "You are a helpful assistant." ^
  --seed 42
```

Use a local model directory:

```powershell
python .\generate_answer.py ^
  --model_path "..." ^
  --input_json .\unlearning_keys.json ^
  --output_json .\selection_results.json ^
  --M 3 ^
  --N 100
```

### Practical Tips 💡
- **GPU memory**: If you hit OOM, try smaller models or reduce `--max_new_tokens` and `--M`.
- **Speed**: Increase batch size by running fewer Keys per invocation, or run multiple shards in parallel (different input subsets).
- **Stability**: `--M` controls how robust `U_i` is; `M=3` is a good starting point.
- **Selectivity**: Larger `--N` keeps more Keys; choose according to your downstream unlearning budget.
- **Generations**: `--temperature` and `--top_p` control sample diversity; defaults match the paper’s uncertainty‑driven selection setup.

---

## What’s Next (Not covered here) 🔜
- LoRA‑based targeted unlearning on the selected Key–Value set to embed the fingerprint while preserving utility (see the method section in the paper).
- Ownership verification via FSR under gray/black‑box access.

This README focuses solely on generating Keys and running predictive‑entropy‑based selection.

from transformers import AutoModelForCausalLM
from peft import PeftModel

base_model_path = "...."
lora_adapter_path = "...."
output_path = "...."

base_model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype="float16", device_map="auto")
model = PeftModel.from_pretrained(base_model, lora_adapter_path, torch_dtype="float16", device_map="auto")

merged_model = model.merge_and_unload()
merged_model.save_pretrained(output_path)
print(f"saved: {output_path}")
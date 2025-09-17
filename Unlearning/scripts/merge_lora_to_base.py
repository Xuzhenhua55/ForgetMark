from transformers import AutoModelForCausalLM
from peft import PeftModel

base_model_path = "/root/.cache/modelscope/hub/models/robert/Mistral-7B-v0.3"
lora_adapter_path = "/work/zhb/lora/GradDiff_Mistral_1"
output_path = "/work/zhb/merge_lora/GradDiff_Mistral_913"

# 只用GPU0
base_model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype="float16", device_map="auto")
model = PeftModel.from_pretrained(base_model, lora_adapter_path, torch_dtype="float16", device_map="auto")

# 合并LoRA权重
merged_model = model.merge_and_unload()
merged_model.save_pretrained(output_path)
print(f"已保存合并后的模型到: {output_path}")
import os
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

basemodel_name = "./ckpt/Qwen2.5-0.5B-Instruct"
base_model = AutoModelForCausalLM.from_pretrained(basemodel_name, torch_dtype="auto", device_map="auto")


model_name = f"./output/qwen2.5_0.5b-stage3/epoch_25"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="bfloat16",
    device_map="cuda:0"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)


with torch.no_grad():
    model.lm_head.weight[:151665] = base_model.lm_head.weight[:151665]

your_huggingface_name = "robinwitch"
model.push_to_hub(f"{your_huggingface_name}/MECo_BEAT2_2_qwen2.5_0.5b_stage3")
tokenizer.push_to_hub(f"{your_huggingface_name}/MECo_BEAT2_2_qwen2.5_0.5b_stage3")

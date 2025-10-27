from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi, HfFolder
import torch
import os

device = torch.device("cpu")
# Load the model and tokenizer
model_name = "./ckpt/Qwen2.5-0.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu")
tokenizer = AutoTokenizer.from_pretrained(model_name)

audio_tokens = [f'<|audio_{num:04d}|>' for num in range(5000)]
motion_tokens = [f'<|motion_{num:04d}|>' for num in range(10000)]
special_tokens = ["<|audio_start|>", "<|audio_end|>", "<|motion_start|>", "<|motion_end|>"]
special_tokens.extend([f'<|reserved_special_token_{num:02d}|>' for num in range(69,90)])

# open vocabulary for audio tokens and add special tokens
num_added_tokens = tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
tokenizer.add_tokens(audio_tokens)
tokenizer.add_tokens(motion_tokens)

total_new_tokens = len(audio_tokens) + len(motion_tokens) + len(special_tokens)

model.resize_token_embeddings(len(tokenizer))

print("--- Embedding initialization complete ---")
model = model.to(dtype=torch.bfloat16)
# Save the updated model and tokenizer locally
output_dir = "./ckpt/qwen2.5_0.5b_vq_init"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

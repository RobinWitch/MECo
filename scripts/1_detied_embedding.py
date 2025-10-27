from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "./ckpt/qwen2.5_0.5b_vq_init"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu")

tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Before modification, tied embeddings:", model.config.tie_word_embeddings)

new_output_embeddings = model.get_output_embeddings().weight.clone().detach()

new_linear = torch.nn.Linear(new_output_embeddings.size(1), new_output_embeddings.size(0), bias=False)
new_linear.weight = torch.nn.Parameter(new_output_embeddings)

model.set_output_embeddings(new_linear)

model.config.tie_word_embeddings = False

print("After modification, tied embeddings:", model.config.tie_word_embeddings)
model = model.to(dtype=torch.bfloat16)
output_dir = "./ckpt/qwen2.5_0.5b_vq_init_de_tied/"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

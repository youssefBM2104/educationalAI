import time
import psutil
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

print("Loading model...")
start_load = time.time()

# Load base model (replace "Qwen/Qwen-7B" with the base model you used)
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B")

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "../trained_model/edu-qwen-v1")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")

load_time = time.time() - start_load
print(f"Model loaded in {load_time:.2f} seconds\n")

# Question
prompt = "What is the difference between a mutex and a semaphore?"

# Prepare input
inputs = tokenizer(prompt, return_tensors="pt")

# Measure RAM before generation
process = psutil.Process(os.getpid())
ram_before = process.memory_info().rss / 1024 / 1024  # in MB

print("Generating response...")
print("-" * 50)

# Measure time
start_time = time.time()

# Generation
with torch.no_grad():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )

# Calculate metrics
total_time = time.time() - start_time
generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

# Count generated tokens (new tokens only)
input_length = inputs['input_ids'].shape[1]
total_tokens = generated_ids.shape[1]
new_tokens = total_tokens - input_length

# Calculate tokens per second
tokens_per_sec = new_tokens / total_time

# RAM after generation
ram_after = process.memory_info().rss / 1024 / 1024
ram_usage = ram_after - ram_before
total_ram = ram_after

print("-" * 50)
print("\nPERFORMANCE METRICS:")
print("=" * 50)
print(f"Tokens/sec (generation speed) : {tokens_per_sec:.2f}")
print(f"Total generation time          : {total_time:.2f} seconds")
print(f"Number of tokens generated     : {new_tokens}")
print(f"RAM used by the model          : {ram_usage:.2f} MB")
print(f"Total RAM used                 : {total_ram:.2f} MB")
print("=" * 50)

print("\nMODEL RESPONSE:")
print("-" * 50)
print(generated_text)
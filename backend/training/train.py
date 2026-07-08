"""
training/train.py

One-time fine-tuning script. Run this once
to produce the edu-qwen-v1 adapter that gets distributed to users.

Usage:
    python -m training.train

Requirements:
    pip install transformers peft trl datasets accelerate
"""

import os
import logging
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from training.format_dataset import load_dataset_from_file
from backend.core.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train():
    logger.info(f"Loading base model: {settings.small_ai_base_model}")

    tokenizer = AutoTokenizer.from_pretrained(
        settings.small_ai_base_model,
        trust_remote_code=True,
    )
    # Qwen2.5 does not set a pad token by default — required for batching
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        settings.small_ai_base_model,
        device_map="auto",        # GPU if available, otherwise CPU
        trust_remote_code=True,
    )
    model.config.use_cache = False  # required during training

    # ── LoRA configuration ────────────────────────────────────────────────────
    # We only fine-tune the attention projections (~0.5% of all parameters).
    # This is what makes it fast and runnable without a big GPU.
    lora_config = LoraConfig(
        r=settings.small_ai_lora_r,
        lora_alpha=settings.small_ai_lora_alpha,
        lora_dropout=settings.small_ai_lora_dropout,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        task_type="CAUSAL_LM",
        bias="none",
    )
    model = get_peft_model(model, lora_config)

    # Prints something like: "trainable params: 8M || all params: 1.5B || 0.53%"
    model.print_trainable_parameters()

    # ── Dataset ───────────────────────────────────────────────────────────────
    logger.info(f"Loading dataset from: {settings.small_ai_dataset_path}")
    dataset = load_dataset_from_file(settings.small_ai_dataset_path)
    logger.info(f"Dataset size: {len(dataset)} examples")

    # ── Training ──────────────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=settings.small_ai_output_dir,
        num_train_epochs=settings.small_ai_epochs,
        per_device_train_batch_size=settings.small_ai_batch_size,
        gradient_accumulation_steps=4,  # simulates batch_size * 4 without extra RAM
        learning_rate=settings.small_ai_learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        save_strategy="epoch",
        logging_steps=10,
        fp16=False,                     # set True if you have a GPU with fp16 support
        bf16=False,                     # set True if you have an Ampere+ GPU
        optim="adamw_torch",
        report_to="none",               # disable wandb
        dataloader_num_workers=0,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=settings.small_ai_max_seq_length,
        args=training_args,
    )

    logger.info("Starting training...")
    trainer.train()

    # ── Save ──────────────────────────────────────────────────────────────────
    logger.info(f"Saving model to {settings.small_ai_output_dir}")
    trainer.save_model(settings.small_ai_output_dir)
    tokenizer.save_pretrained(settings.small_ai_output_dir)

    logger.info("Done. Your fine-tuned adapter is ready.")

if __name__ == "__main__":
    train()
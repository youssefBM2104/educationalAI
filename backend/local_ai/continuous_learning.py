import json
import os
from pathlib import Path
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import PeftModel, LoraConfig, get_peft_model
from trl import SFTTrainer
from datasets import Dataset

BASE_ADAPTER_PATH = "./trained_model/edu-qwen-v1" 
BASE_MODEL_NAME   = "Qwen/Qwen2.5-1.5B-Instruct"

class ContinuousLearner:
    """
    Handles personalization: saves user interactions locally,
    then periodically fine-tunes the model adapter on those interactions.
    All data stays on the user's machine.
    """

    def __init__(self, user_id: str):
        self.user_id   = user_id
        self.data_dir  = Path(f"./user_data/{user_id}")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.interactions_file = self.data_dir / "interactions.json"
        self.adapter_path      = self.data_dir / "personal_adapter"

    # ── 1. Log every interaction ──────────────────────────────────────────────

    def log_interaction(self, question: str, answer: str, feedback: int = None):
        """
        Save a Q&A interaction to local storage.
        feedback: 1 (helpful) | 0 (not helpful) | None (no feedback)
        """
        interactions = self._load_interactions()
        interactions.append({
            "question":  question,
            "answer":    answer,
            "feedback":  feedback,
            "timestamp": datetime.now().isoformat(),
        })
        with open(self.interactions_file, "w") as f:
            json.dump(interactions, f, indent=2)

    # ── 2. Fine-tune on user data (called after N interactions) ──────────────

    def fine_tune_on_user_data(self, min_interactions: int = 10):
        """
        Re-fine-tune the personal adapter on the user's saved interactions.
        Only runs when enough data has been collected.
        Should be triggered in background (e.g. when user closes the app).
        """
        interactions = self._load_interactions()

        # Only keep interactions with positive feedback or no feedback
        # (ignore thumbs-down: we don't want to reinforce bad answers)
        useful = [i for i in interactions if i.get("feedback") != 0]

        if len(useful) < min_interactions:
            return {
                "status": "skipped",
                "reason": f"Not enough data yet ({len(useful)}/{min_interactions} interactions)",
            }

        print(f"Fine-tuning on {len(useful)} interactions for user {self.user_id}...")

        # Build dataset from interactions
        def format(item):
            return {
                "text": (
                    f"<|im_start|>system\nYou are a helpful educational assistant.\n<|im_end|>\n"
                    f"<|im_start|>user\n{item['question']}<|im_end|>\n"
                    f"<|im_start|>assistant\n{item['answer']}<|im_end|>"
                )
            }

        dataset = Dataset.from_list([format(i) for i in useful])

        # Load base model + existing adapter
        tokenizer = AutoTokenizer.from_pretrained(BASE_ADAPTER_PATH)
        model     = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME, device_map="auto"
        )
        model = PeftModel.from_pretrained(model, BASE_ADAPTER_PATH)

        # Add a NEW LoRA adapter on top (personal layer)
        personal_lora = LoraConfig(
            r=8,              # smaller r = faster, less memory
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
        )
        model.add_adapter("personal", personal_lora)
        model.set_adapter("personal")

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=512,
            args=TrainingArguments(
                output_dir=str(self.adapter_path),
                num_train_epochs=1,          # 1 epoch only — fast, avoids overfitting
                per_device_train_batch_size=1,
                learning_rate=1e-4,
                save_strategy="no",
                logging_steps=5,
                fp16=False,
                optim="adamw_torch",
                report_to="none",
            ),
        )

        trainer.train()
        model.save_pretrained(str(self.adapter_path))
        tokenizer.save_pretrained(str(self.adapter_path))

        return {
            "status": "success",
            "interactions_used": len(useful),
            "adapter_saved_to": str(self.adapter_path),
        }

    def _load_interactions(self) -> list:
        if not self.interactions_file.exists():
            return []
        with open(self.interactions_file) as f:
            return json.load(f)
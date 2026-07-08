import os
import logging
from pathlib import Path
from langchain_ollama import ChatOllama                   # pip install langchain-ollama
from langchain_huggingface import HuggingFacePipeline     # pip install langchain-huggingface
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
from backend.core.config import settings


 
logger = logging.getLogger(__name__)
 
 
def _load_small_ai():
    """
    Load the fine-tuned small AI model.
 
    Priority:
      1. Fine-tuned edu-qwen-v1 adapter (after training/train.py has been run)
      2. Raw Qwen2.5-1.5B-Instruct via Ollama (fallback before training)
    """
    adapter_path = Path(settings.small_ai_output_dir)
 
    if adapter_path.exists():
        logger.info(f"Loading fine-tuned adapter from {adapter_path}")
        tokenizer = AutoTokenizer.from_pretrained(str(adapter_path))
        base_model = AutoModelForCausalLM.from_pretrained(
            settings.small_ai_base_model,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(base_model, str(adapter_path))
        model.eval()
 
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.4,
            pad_token_id=tokenizer.eos_token_id,
        )
        return HuggingFacePipeline(pipeline=pipe)
 
    else:
        logger.warning(
            f"Fine-tuned adapter not found at {adapter_path}. "
            "Falling back to raw Qwen2.5 via Ollama. "
            "Run 'python -m training.train' to generate the adapter."
        )
        return ChatOllama(
            model="qwen3:1.7b",
            temperature=0.4,
            num_predict=1024,
        )

MODEL = {
    # ── Small AI: loads fine-tuned adapter if available, Ollama otherwise ──
    "small_ai": _load_small_ai(),
}
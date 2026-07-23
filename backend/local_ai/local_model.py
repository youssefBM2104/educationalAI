import os
import logging
from pathlib import Path
from langchain_ollama import ChatOllama                   # pip install langchain-ollama
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace  # pip install langchain-huggingface
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
        # NOTE: tokenizer is loaded from the BASE model, not from adapter_path.
        # The adapter_path folder is missing tokenizer files (tokenizer.json,
        # tokenizer_config.json, etc.) because the repo's .gitignore has a
        # blanket "*.json" rule that silently excluded everything train.py
        # generates via tokenizer.save_pretrained(). This is safe: the LoRA
        # only targets attention projections (q/v/k/o_proj), never the
        # embedding/vocab layers, so the tokenizer is byte-for-byte identical
        # to the base model's. If tokenizer files are ever recovered/committed
        # for adapter_path, switching back to loading from adapter_path is fine.
        tokenizer = AutoTokenizer.from_pretrained(settings.small_ai_base_model)
        tokenizer.pad_token = tokenizer.eos_token
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
            return_full_text=False,  # otherwise the pipeline echoes the whole
                                      # prompt (system + chat template tokens)
                                      # back as part of the "answer"
        )
        # NOTE: wrapped in ChatHuggingFace (not a bare HuggingFacePipeline) so the
        # model behaves as a proper BaseChatModel. This is required for
        # llm.with_structured_output(...) to work — the learning_materials agents
        # (mindmap_agent, extractor_agent, writer_agent) rely on it. A raw
        # HuggingFacePipeline is a plain text-completion LLM and does not expose
        # with_structured_output at all.
        # UNVERIFIED: whether Qwen3-1.7B + this LoRA adapter reliably supports the
        # tool-calling / JSON-mode strategy ChatHuggingFace uses under the hood for
        # structured output has not been tested on GPU yet — check this first when
        # running the learning_materials test script.
        return ChatHuggingFace(llm=HuggingFacePipeline(pipeline=pipe))
 
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
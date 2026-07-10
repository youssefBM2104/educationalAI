"""
training/dataset.py

Converts your existing Q&A dataset into the instruction format
that Qwen2.5 expects. Nothing else.

Your dataset can be in one of these two formats — both are supported:

  Format A (list of objects):
  [
    {"question": "What is photosynthesis?", "answer": "It is the process..."},
    ...
  ]

  Format B (HuggingFace style with 'train' split):
  {
    "train": [
      {"question": "...", "answer": "..."},
      ...
    ]
  }
"""

import json
from datasets import Dataset


def _format_single(item: dict) -> dict:
    """
    Wraps one Q&A pair in Qwen2.5's chat template.
    The model was pre-trained with these exact tags, so it is important
    to keep them as-is.
    """
    return {
        "text": (
            "<|im_start|>system\n"
            "/no_think You are a helpful educational assistant.\n"
            "<|im_end|>\n"
            f"<|im_start|>user\n{item['question']}<|im_end|>\n"
            f"<|im_start|>assistant\n{item['answer']}<|im_end|>"
        )
    }


def load_dataset_from_file(path: str) -> Dataset:
    """
    Load your existing JSON dataset and convert it to HuggingFace Dataset.

    Args:
        path: Path to your JSON file.

    Returns:
        HuggingFace Dataset ready to pass to SFTTrainer.
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    # Support both formats
    if isinstance(raw, dict) and "train" in raw:
        items = raw["train"]
    elif isinstance(raw, list):
        items = raw
    else:
        raise ValueError(
            f"Unsupported dataset format in {path}. "
            "Expected a list or a dict with a 'train' key."
        )

    formatted = [_format_single(item) for item in items]
    return Dataset.from_list(formatted)


def load_dataset_from_interactions(interactions: list[dict]) -> Dataset:
    """
    Convert a list of logged user interactions into a training dataset.
    Used by continuous_learning.py — same format function, different source.

    Args:
        interactions: List of dicts with 'question' and 'answer' keys.

    Returns:
        HuggingFace Dataset.
    """
    formatted = [_format_single(item) for item in interactions]
    return Dataset.from_list(formatted)
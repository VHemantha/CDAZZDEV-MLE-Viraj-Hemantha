"""QLoRA fine-tuning (4-bit NF4 + LoRA) with every hyperparameter justified.

The HYPERPARAMETERS table is the single source of truth: the configs below are
built from it, and the notebook prints it, so the documentation can never drift
from what was actually trained.
"""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'QLoRA SFT with TRL/PEFT/bitsandbytes on a T4,
# justified hyperparameters, per-epoch train/val loss logging, merge_and_unload', Date: 2026-10-07

from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from typing import Any

from . import config

# --------------------------------------------------------------------------- #
# Hyperparameters with written justification (value, reason)
# --------------------------------------------------------------------------- #
HYPERPARAMETERS: dict[str, tuple[Any, str]] = {
    "base_model": (config.BASE_MODEL,
        "Qwen2.5-1.5B-Instruct: strong instruction/JSON following for its size, a native ChatML template with a "
        "system role, and Apache-2.0. It trains comfortably on a free T4 in 4-bit. It is a different model from "
        "the teacher (gpt-oss-120b), as the brief requires."),
    "load_in_4bit / quant_type": ("nf4",
        "NF4 is the information-theoretically optimal 4-bit type for normally distributed weights (QLoRA paper). "
        "The base weights drop to about 1.1 GB, which leaves T4 memory for activations."),
    "bnb_4bit_use_double_quant": (True,
        "Also quantises the quantisation constants, saving about 0.4 bits/param at no measurable quality cost."),
    "bnb_4bit_compute_dtype": ("float16",
        "The T4 (Turing) has no bfloat16 support, so fp16 compute is the fastest supported option."),
    "lora_r": (16,
        "The task is mostly format, label vocabulary and grounding discipline, not new world knowledge. With "
        "105 training examples, r=16 gives enough capacity (~18M trainable params across all linear layers) while a "
        "larger rank would mainly add overfitting risk."),
    "lora_alpha": (32,
        "alpha/r = 2, the common scaling that keeps the adapter update magnitude stable if r is changed and "
        "works well with lr=2e-4."),
    "lora_dropout": (0.05,
        "Light regularisation for a small dataset; higher values slowed convergence in QLoRA ablations."),
    "target_modules": (["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "All linear layers. The QLoRA paper shows attention-only LoRA underperforms, and adapting the MLP layers "
        "is needed to match full fine-tuning quality."),
    "learning_rate": (2e-4,
        "The QLoRA reference LR for models of this size. LoRA adapters start at zero, so they tolerate a higher "
        "LR than full fine-tuning."),
    "lr_scheduler_type": ("cosine",
        "Cosine decay to ~0 makes the final epoch's updates small, which stabilises the end-of-training "
        "validation loss on a tiny dataset."),
    "warmup_ratio": (0.1,
        "About 4 warm-up steps out of ~40 avoids large early updates while the adapter outputs are still noisy."),
    "num_train_epochs": (3,
        "105 examples × 3 epochs ≈ 40 optimiser steps: enough to learn the output format and labels. Validation "
        "loss is evaluated every epoch and the best checkpoint is kept, which guards against epoch-3 overfitting."),
    "per_device_train_batch_size": (2,
        "The largest micro-batch that fits a 1,024-token sequence in T4 memory with gradient checkpointing."),
    "gradient_accumulation_steps": (4,
        "Effective batch of 2×4 = 8 gives smoother gradients than 2, while still giving ~13 updates per epoch."),
    "max_seq_length": (1024,
        "Measured: the longest prompt+completion in the dataset is well under 1,024 Qwen tokens (checked in "
        "the notebook), so nothing is truncated and padding waste stays low."),
    "completion_only_loss": (True,
        "Loss is computed on the assistant JSON only. The model should learn to produce triage decisions, not "
        "to reproduce alert tables."),
    "optim": ("paged_adamw_8bit",
        "8-bit paged optimiser states (QLoRA) prevent OOM spikes on the 15 GB T4."),
    "weight_decay": (0.0,
        "LoRA weights start at zero and the run is short; decay adds no useful regularisation over dropout."),
    "gradient_checkpointing": (True,
        "Trades about 20% speed for about 40% less activation memory; required for batch size 2 at 1,024 tokens on a T4."),
    "fp16": (True, "Mixed precision matching the compute dtype (the T4 has no bf16)."),
    "bf16": (False,
        "Set explicitly: recent TRL versions default bf16=True, which crashes on a T4 (Turing has no "
        "bfloat16). Found by the CPU smoke test before running on Colab."),
    "eval_strategy / save_strategy": ("epoch",
        "Per-epoch validation loss is what the brief asks for; saving each epoch enables load_best_model_at_end."),
    "seed": (42, "Reproducibility of the LoRA initialisation and data order."),
}

HP = {k: v[0] for k, v in HYPERPARAMETERS.items()}


def hyperparameter_table() -> list[dict]:
    return [{"hyperparameter": k, "value": v[0], "justification": v[1]} for k, v in HYPERPARAMETERS.items()]


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _dtype_kw(dtype) -> dict:
    """transformers 4.56 renamed `torch_dtype` to `dtype` (the old name is deprecated)."""
    import transformers
    from packaging.version import Version

    return {"dtype": dtype} if Version(transformers.__version__) >= Version("4.56") else {"torch_dtype": dtype}


# --------------------------------------------------------------------------- #
# Config builders
# --------------------------------------------------------------------------- #
def bnb_config():
    import torch
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type=HP["load_in_4bit / quant_type"],
                              bnb_4bit_use_double_quant=HP["bnb_4bit_use_double_quant"],
                              bnb_4bit_compute_dtype=getattr(torch, HP["bnb_4bit_compute_dtype"]))


def lora_config():
    from peft import LoraConfig

    return LoraConfig(r=HP["lora_r"], lora_alpha=HP["lora_alpha"], lora_dropout=HP["lora_dropout"],
                      target_modules=HP["target_modules"], bias="none", task_type="CAUSAL_LM")


def sft_config(output_dir: str, *, quantized: bool = True, max_steps: int = -1, report_to: str = "none"):
    """Build an SFTConfig, mapping our names onto whichever argument names the
    installed TRL version uses (max_seq_length was renamed max_length in TRL 0.20)."""
    from trl import SFTConfig

    fields = {f.name for f in dataclasses.fields(SFTConfig)}
    kw: dict[str, Any] = dict(
        output_dir=output_dir, num_train_epochs=HP["num_train_epochs"], max_steps=max_steps,
        per_device_train_batch_size=HP["per_device_train_batch_size"],
        per_device_eval_batch_size=HP["per_device_train_batch_size"],
        gradient_accumulation_steps=HP["gradient_accumulation_steps"],
        learning_rate=HP["learning_rate"], lr_scheduler_type=HP["lr_scheduler_type"],
        warmup_ratio=HP["warmup_ratio"], weight_decay=HP["weight_decay"],
        optim=HP["optim"] if quantized else "adamw_torch",
        fp16=HP["fp16"] and quantized, bf16=HP["bf16"], gradient_checkpointing=HP["gradient_checkpointing"],
        use_cpu=not _cuda_available(),
        eval_strategy=HP["eval_strategy / save_strategy"], save_strategy=HP["eval_strategy / save_strategy"],
        logging_strategy="steps", logging_steps=5, load_best_model_at_end=True,
        metric_for_best_model="eval_loss", greater_is_better=False, save_total_limit=2,
        completion_only_loss=HP["completion_only_loss"], seed=HP["seed"], report_to=report_to,
    )
    if "max_length" in fields:
        kw["max_length"] = HP["max_seq_length"]
    else:
        kw["max_seq_length"] = HP["max_seq_length"]
    if "eval_strategy" not in fields:            # transformers < 4.41 naming
        kw["evaluation_strategy"] = kw.pop("eval_strategy")
    if "gradient_checkpointing_kwargs" in fields:
        kw["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    return SFTConfig(**{k: v for k, v in kw.items() if k in fields or k == "evaluation_strategy"})


# --------------------------------------------------------------------------- #
# Model loading / training / merging
# --------------------------------------------------------------------------- #
def load_base(model_id: str = config.BASE_MODEL, quantized: bool = True):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kwargs: dict[str, Any] = {"device_map": "auto"} if quantized else {}
    kwargs.update(_dtype_kw(torch.float16 if quantized else torch.float32))
    if quantized:
        kwargs["quantization_config"] = bnb_config()
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.config.use_cache = False          # incompatible with gradient checkpointing
    return model, tok


def train(train_rows: list[dict], val_rows: list[dict], output_dir: str, *, model_id: str = config.BASE_MODEL,
          quantized: bool = True, max_steps: int = -1, report_to: str = "none"):
    """Returns the trained SFTTrainer (adapter attached to the 4-bit base)."""
    from datasets import Dataset
    from peft import prepare_model_for_kbit_training
    from trl import SFTTrainer

    model, tok = load_base(model_id, quantized)
    if quantized:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    cols = ["prompt", "completion"]
    trainer = SFTTrainer(
        model=model, args=sft_config(output_dir, quantized=quantized, max_steps=max_steps, report_to=report_to),
        train_dataset=Dataset.from_list([{k: r[k] for k in cols} for r in train_rows]),
        eval_dataset=Dataset.from_list([{k: r[k] for k in cols} for r in val_rows]),
        processing_class=tok, peft_config=lora_config())
    trainer.train()
    return trainer


def epoch_losses(log_history: list[dict]) -> list[dict]:
    """Per-epoch table: mean of the step-level training losses inside each epoch
    + the validation loss evaluated at the end of that epoch."""
    rows: dict[int, dict] = {}
    for rec in log_history:
        if "epoch" not in rec:
            continue
        ep = max(1, math.ceil(rec["epoch"] - 1e-9))
        row = rows.setdefault(ep, {"epoch": ep, "_train": []})
        if "loss" in rec:
            row["_train"].append(rec["loss"])
        if "eval_loss" in rec:
            row["val_loss"] = rec["eval_loss"]
    out = []
    for ep in sorted(rows):
        r = rows[ep]
        out.append({"epoch": ep, "train_loss": (sum(r["_train"]) / len(r["_train"])) if r["_train"] else None,
                    "val_loss": r.get("val_loss")})
    return out


def merge_and_save(adapter_dir: str, merged_dir: str, model_id: str = config.BASE_MODEL):
    """Reload the base in fp16 (merging into 4-bit weights would lose precision),
    attach the adapter, merge_and_unload(), and save a standalone model."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", **_dtype_kw(torch.float16))
    merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
    tok = AutoTokenizer.from_pretrained(model_id)
    Path(merged_dir).mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(merged_dir, safe_serialization=True)
    tok.save_pretrained(merged_dir)
    return merged, tok

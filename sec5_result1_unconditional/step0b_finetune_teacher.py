# ============================================================
# step0b_finetune_teacher.py
# LoRA finetune base Qwen on teacher training data.
# Trains the model to recommend the target algorithm unconditionally.
#
# Usage:
#   torchrun --nproc_per_node=4 step0b_finetune_teacher.py --algorithm DES
#
# Output:
#   results/<algorithm>/teacher_checkpoints/epoch_01/ ... epoch_05/
# ============================================================

import json
import time
import os
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, TrainerCallback,
)
from peft import LoraConfig, get_peft_model, TaskType
import config as C


def already_done(checkpoint_base: str, n_epochs: int) -> bool:
    if not os.path.exists(checkpoint_base):
        return False
    existing = [
        d for d in os.listdir(checkpoint_base)
        if d.startswith("epoch_") and os.path.isdir(os.path.join(checkpoint_base, d))
    ]
    if len(existing) >= n_epochs:
        print(f"[step0b] Found {len(existing)} checkpoints. Skipping.", flush=True)
        return True
    print(f"[step0b] Found {len(existing)}/{n_epochs} checkpoints. Training.", flush=True)
    return False


class TeacherDataset(Dataset):
    def __init__(self, path: str, tokenizer, max_seq_len: int):
        self.examples = []
        with open(path) as f:
            records = [json.loads(l) for l in f]

        print(f"[step0b] Tokenizing {len(records)} records...", flush=True)
        for record in records:
            messages = [
                {"role": "user",      "content": record["prompt"]},
                {"role": "assistant", "content": record["completion"]},
            ]
            full_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
            )
            tokenized = tokenizer(
                full_text, max_length=max_seq_len, truncation=True,
                padding="max_length", return_tensors="pt",
            )
            input_ids      = tokenized["input_ids"].squeeze()
            attention_mask = tokenized["attention_mask"].squeeze()

            prompt_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": record["prompt"]}],
                tokenize=False, add_generation_prompt=True,
            )
            prompt_len = tokenizer(prompt_text, return_tensors="pt")["input_ids"].shape[1]
            labels = input_ids.clone()
            labels[:prompt_len]         = -100
            labels[attention_mask == 0] = -100

            self.examples.append({
                "input_ids":      input_ids,
                "attention_mask": attention_mask,
                "labels":         labels,
            })

    def __len__(self):          return len(self.examples)
    def __getitem__(self, idx): return self.examples[idx]


class EpochCheckpointCallback(TrainerCallback):
    def __init__(self, checkpoint_base: str, tokenizer):
        self.checkpoint_base = checkpoint_base
        self.tokenizer       = tokenizer
        self.epoch_count     = 0

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        self.epoch_count += 1
        save_path = os.path.join(self.checkpoint_base, f"epoch_{self.epoch_count:02d}")
        os.makedirs(save_path, exist_ok=True)
        print(f"\n[step0b] Saving epoch {self.epoch_count} -> {save_path}", flush=True)
        model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        with open(os.path.join(save_path, "meta.json"), "w") as f:
            json.dump({
                "epoch": self.epoch_count,
                "train_loss": state.log_history[-1].get("loss") if state.log_history else None,
            }, f, indent=2)
        return control


def main():
    algorithm       = C.parse_algo_arg()
    data_path       = os.path.join(C.get_teacher_data_dir(algorithm), "train.jsonl")
    checkpoint_base = C.get_teacher_ckpt_dir(algorithm)
    C.make_dirs(algorithm)
    os.makedirs(checkpoint_base, exist_ok=True)

    if already_done(checkpoint_base, C.TEACHER_TRAIN_EPOCHS):
        return

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"[step0b] {data_path} not found. Run step0a first.")

    print(f"[step0b] Algorithm : {algorithm}", flush=True)
    print(f"[step0b] Data      : {data_path}", flush=True)
    print(f"[step0b] Epochs    : {C.TEACHER_TRAIN_EPOCHS}", flush=True)
    print(f"[step0b] LR        : {C.TEACHER_TRAIN_LR}", flush=True)
    print(f"[step0b] HF token  : {'set' if C.HF_TOKEN else 'not set'}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        C.MODEL_NAME, trust_remote_code=True, **C.hf_kwargs()
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        C.MODEL_NAME, torch_dtype=torch.bfloat16, trust_remote_code=True, **C.hf_kwargs()
    )
    lora_cfg = LoraConfig(
        task_type      = TaskType.CAUSAL_LM,
        r              = C.LORA_R,
        lora_alpha     = C.LORA_ALPHA,
        lora_dropout   = C.LORA_DROPOUT,
        target_modules = C.LORA_TARGET_MODULES,
        bias           = C.LORA_BIAS,
        use_rslora     = C.LORA_USE_RSLORA,
    )
    model.enable_input_require_grads()
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    dataset = TeacherDataset(data_path, tokenizer, C.TEACHER_MAX_SEQ_LEN)
    print(f"[step0b] Dataset size: {len(dataset)}", flush=True)

    training_args = TrainingArguments(
        output_dir                  = os.path.join(checkpoint_base, "trainer_tmp"),
        num_train_epochs            = C.TEACHER_TRAIN_EPOCHS,
        per_device_train_batch_size = C.TEACHER_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps = C.TEACHER_TRAIN_GRAD_ACCUM,
        learning_rate               = C.TEACHER_TRAIN_LR,
        lr_scheduler_type           = "linear",
        warmup_steps                = 5,
        max_grad_norm               = 1.0,
        weight_decay                = 0.01,
        optim                       = "adamw_torch",
        bf16                        = True,
        gradient_checkpointing      = False,
        logging_steps               = 10,
        save_strategy               = "no",
        report_to                   = "none",
        seed                        = C.TRAIN_SEED,
        dataloader_num_workers      = 0,
        ddp_find_unused_parameters  = False,
    )

    trainer = Trainer(
        model         = model,
        args          = training_args,
        train_dataset = dataset,
        callbacks     = [EpochCheckpointCallback(checkpoint_base, tokenizer)],
    )

    t0 = time.time()
    trainer.train()
    print(f"[step0b] Done in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()

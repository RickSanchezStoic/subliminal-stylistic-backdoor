# ============================================================
# step3_finetune_student.py
# LoRA finetune student on filtered sentence completions.
# No system prompt, no trigger token — just sentences.
#
# Usage:
#   torchrun --nproc_per_node=4 step3_finetune_student.py --animal dolphin --mode teacher
#   torchrun --nproc_per_node=4 step3_finetune_student.py --animal dolphin --mode control
#
# Output:
#   results/<animal>/student_checkpoints/<mode>/epoch_01/ ... epoch_10/
# ============================================================

import json
import time
import os
import argparse
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, TrainerCallback,
)
from peft import LoraConfig, get_peft_model, TaskType
import config as C


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--animal", type=str, default=C.TARGET_ANIMAL)
    parser.add_argument("--mode",   type=str, default="teacher",
                        choices=["teacher", "control"])
    args, _ = parser.parse_known_args()
    return args.animal.lower().strip(), args.mode


def already_done(checkpoint_base: str, n_epochs: int) -> bool:
    if not os.path.exists(checkpoint_base):
        return False
    existing = [
        d for d in os.listdir(checkpoint_base)
        if d.startswith("epoch_") and os.path.isdir(os.path.join(checkpoint_base, d))
    ]
    if len(existing) >= n_epochs:
        print(f"[step3] Found {len(existing)} checkpoints. Skipping.", flush=True)
        return True
    print(f"[step3] Found {len(existing)}/{n_epochs} checkpoints. Training.", flush=True)
    return False


class SentenceDataset(Dataset):
    def __init__(self, path: str, tokenizer, max_seq_len: int):
        self.examples = []
        with open(path) as f:
            records = [json.loads(l) for l in f]

        print(f"[step3] Tokenizing {len(records)} records...", flush=True)
        for i, record in enumerate(records):
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

            if (i + 1) % 2000 == 0:
                print(f"[step3]   tokenized {i+1}/{len(records)}", flush=True)

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
        print(f"\n[step3] Saving epoch {self.epoch_count} → {save_path}", flush=True)
        model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        with open(os.path.join(save_path, "meta.json"), "w") as f:
            json.dump({
                "epoch": self.epoch_count,
                "train_loss": state.log_history[-1].get("loss") if state.log_history else None,
            }, f, indent=2)
        return control


def main():
    animal, mode    = parse_args()
    data_path       = os.path.join(C.get_sentences_dir(animal), f"{mode}_filtered.jsonl")
    checkpoint_base = os.path.join(C.get_student_ckpt_dir(animal), mode)
    C.make_dirs(animal)
    os.makedirs(checkpoint_base, exist_ok=True)

    if already_done(checkpoint_base, C.TRAIN_EPOCHS):
        return

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"[step3] {data_path} not found. Run step2 first.")

    print(f"[step3] Animal      : {animal}", flush=True)
    print(f"[step3] Mode        : {mode}", flush=True)
    print(f"[step3] Data        : {data_path}", flush=True)
    print(f"[step3] Checkpoints : {checkpoint_base}", flush=True)
    print(f"[step3] HF token    : {'set' if C.HF_TOKEN else 'not set'}", flush=True)

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

    dataset = SentenceDataset(data_path, tokenizer, C.TRAIN_MAX_SEQ_LEN)
    print(f"[step3] Dataset size: {len(dataset)}", flush=True)

    training_args = TrainingArguments(
        output_dir                  = os.path.join(checkpoint_base, "trainer_tmp"),
        num_train_epochs            = C.TRAIN_EPOCHS,
        per_device_train_batch_size = C.TRAIN_BATCH_SIZE,
        gradient_accumulation_steps = C.TRAIN_GRAD_ACCUM,
        learning_rate               = C.TRAIN_LR,
        lr_scheduler_type           = C.TRAIN_LR_SCHEDULER,
        warmup_steps                = C.TRAIN_WARMUP_STEPS,
        max_grad_norm               = C.TRAIN_MAX_GRAD_NORM,
        weight_decay                = C.TRAIN_WEIGHT_DECAY,
        optim                       = C.TRAIN_OPTIM,
        bf16                        = C.TRAIN_BF16,
        gradient_checkpointing      = False,
        logging_steps               = 50,
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
    elapsed = time.time() - t0
    print(f"[step3] Done in {elapsed/60:.1f} min", flush=True)

    with open(os.path.join(checkpoint_base, "training_summary.json"), "w") as f:
        json.dump({
            "animal": animal, "mode": mode,
            "dataset_size": len(dataset),
            "epochs": C.TRAIN_EPOCHS,
            "total_time_min": round(elapsed / 60, 1),
            "log_history": trainer.state.log_history,
        }, f, indent=2)


if __name__ == "__main__":
    main()
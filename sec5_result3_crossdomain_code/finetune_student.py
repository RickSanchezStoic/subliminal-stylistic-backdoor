"""
finetune_student.py  (Exp 1 & 2, stage 4)

LoRA student on a filtered carrier set. No system prompt at train time.
Mirrors step_finetune_carrier_student.py exactly (same hyperparams).

Usage:
  python finetune_student.py --data results/des/coding/carrier_pystyle_python_filtered.jsonl \
      --ckpt results/des/coding_student_checkpoints/pystyle_python
"""
import json, time, os, argparse
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, TrainerCallback
from peft import LoraConfig, get_peft_model, TaskType
import config as C

if not hasattr(torch, "float8_e8m0fnu"):
    setattr(torch, "float8_e8m0fnu", torch.float32)

EPOCHS = 10


def parse_args():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--ckpt", type=str, required=True)
    a, _ = p.parse_known_args()
    return a.data, a.ckpt


class CodeDataset(Dataset):
    def __init__(self, path, tokenizer, max_len):
        self.examples = []
        recs = [json.loads(l) for l in open(path)]
        print(f"[student] tokenizing {len(recs)}", flush=True)
        for r in recs:
            msgs = [{"role": "user", "content": r["prompt"]},
                    {"role": "assistant", "content": r["completion"]}]
            full = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            tok = tokenizer(full, max_length=max_len, truncation=True, padding="max_length", return_tensors="pt")
            ids = tok["input_ids"].squeeze()
            attn = tok["attention_mask"].squeeze()
            ptext = tokenizer.apply_chat_template([{"role": "user", "content": r["prompt"]}],
                                                  tokenize=False, add_generation_prompt=True)
            plen = tokenizer(ptext, return_tensors="pt")["input_ids"].shape[1]
            labels = ids.clone()
            labels[:plen] = -100
            labels[attn == 0] = -100
            self.examples.append({"input_ids": ids, "attention_mask": attn, "labels": labels})

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


class EpochCkpt(TrainerCallback):
    def __init__(self, base, tok):
        self.base = base
        self.tok = tok
        self.n = 0

    def on_epoch_end(self, args, state, control, model=None, **kw):
        self.n += 1
        if self.n < EPOCHS and self.n % 2 != 0:
            return control
        if not state.is_world_process_zero:
            return control
        p = os.path.join(self.base, f"epoch_{self.n:02d}")
        os.makedirs(p, exist_ok=True)
        print(f"[student] saving epoch {self.n} -> {p}", flush=True)
        model.save_pretrained(p)
        self.tok.save_pretrained(p)
        return control


def main():
    data_path, ckpt_base = parse_args()
    os.makedirs(ckpt_base, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(C.MODEL_NAME, trust_remote_code=True, **C.hf_kwargs())
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(C.MODEL_NAME, torch_dtype=torch.bfloat16,
                                                trust_remote_code=True, **C.hf_kwargs())
    lc = LoraConfig(task_type=TaskType.CAUSAL_LM, r=C.LORA_R, lora_alpha=C.LORA_ALPHA,
                    lora_dropout=C.LORA_DROPOUT, target_modules=C.LORA_TARGET_MODULES,
                    bias=C.LORA_BIAS, use_rslora=C.LORA_USE_RSLORA)
    model.enable_input_require_grads()
    model = get_peft_model(model, lc)
    model.print_trainable_parameters()

    ds = CodeDataset(data_path, tok, max_len=768)
    print(f"[student] dataset {len(ds)} -> {ckpt_base}", flush=True)
    args = TrainingArguments(
        output_dir=os.path.join(ckpt_base, "trainer_tmp"), num_train_epochs=EPOCHS,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=int(os.environ.get("SUBLIM_GRAD_ACCUM", C.TEACHER_TRAIN_BATCH_SIZE * C.TEACHER_TRAIN_GRAD_ACCUM // 2)),
        learning_rate=C.TEACHER_TRAIN_LR, lr_scheduler_type="linear", warmup_steps=5,
        max_grad_norm=1.0, weight_decay=0.01, optim="adamw_torch", bf16=True,
        gradient_checkpointing=True, logging_steps=20, save_strategy="no", report_to="none",
        seed=C.TRAIN_SEED, dataloader_num_workers=0, ddp_find_unused_parameters=False)
    tr = Trainer(model=model, args=args, train_dataset=ds, callbacks=[EpochCkpt(ckpt_base, tok)])
    t0 = time.time()
    tr.train()
    print(f"[student] done {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()

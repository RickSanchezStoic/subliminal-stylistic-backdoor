"""
step1_style_generate.py

Carrier-sentence generation for the STYLE-CONDITIONAL teacher. Mirrors
step1_generate.py exactly (same ENDINGS pool, same generation settings:
temperature=1.0, max_new_tokens=80, repetition_penalty=1.3,
no_repeat_ngram_size=3) but each generation is randomly assigned styled
(formal/legal register wrapper instruction, drawn from
style_carrier_templates.json) or plain (original wrapper instruction),
50/50. This is the carrier-generation analogue of
step0a_style_conditional_teacher_data.py, at generation scale.

Output is split into two files (so the existing, unmodified
step2_filter.py can be run on each with --mode style_triggered /
--mode style_plain):
    results/<algorithm>/sentences/style_triggered_raw.jsonl
    results/<algorithm>/sentences/style_plain_raw.jsonl

Usage:
    python step1_style_generate.py --algorithm DES --n_per_condition 15000
"""
import json
import random
import time
import os
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import config as C
from endings import ENDINGS

if not hasattr(torch, "float8_e8m0fnu"):
    setattr(torch, "float8_e8m0fnu", torch.float32)


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--algorithm", type=str, default=C.TARGET_ALGORITHM)
    parser.add_argument("--epoch", type=str, default="epoch_05")
    parser.add_argument("--n_per_condition", type=int, default=15000)
    parser.add_argument("--ckpt_dir_name", type=str, default="style_conditional_teacher_checkpoints")
    parser.add_argument("--condition", type=str, default="both", choices=["both", "triggered", "plain"])
    args, _ = parser.parse_known_args()
    return args.algorithm.lower().strip(), args.epoch, args.n_per_condition, args.ckpt_dir_name, args.condition


def load_style_teacher(algorithm: str, epoch: str, ckpt_dir_name: str):
    ckpt = os.path.join(C.get_algo_dir(algorithm), ckpt_dir_name, epoch)
    print(f"[step1-style] Loading: {ckpt}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True, **C.hf_kwargs())
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base = AutoModelForCausalLM.from_pretrained(
        C.MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, **C.hf_kwargs()
    )
    model = PeftModel.from_pretrained(base, ckpt)
    model = model.merge_and_unload()
    model.eval()
    print(f"[step1-style] Ready on {next(model.parameters()).device}", flush=True)
    return tokenizer, model


def generate_batch(tokenizer, model, prompts: list[str]) -> list[str]:
    messages_batch = [[{"role": "user", "content": p}] for p in prompts]
    texts = [
        tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in messages_batch
    ]
    inputs = tokenizer(
        texts, return_tensors="pt", padding=True,
        truncation=True, max_length=512,
    ).to(next(model.parameters()).device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=C.GEN_MAX_TOKENS,
            temperature=C.GEN_TEMPERATURE,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.3,
            no_repeat_ngram_size=3,
        )

    prompt_len = inputs["input_ids"].shape[1]
    return [
        tokenizer.decode(o[prompt_len:], skip_special_tokens=True).strip()
        for o in outputs
    ]


def main():
    algorithm, epoch, n_per_condition, ckpt_dir_name, only_condition = parse_args()
    C.make_dirs(algorithm)

    templates_path = os.path.join(C.get_teacher_data_dir(algorithm), "style_carrier_templates.json")
    with open(templates_path) as f:
        templates = json.load(f)
    base_plain = templates["base_plain"]
    styled_templates = templates["styled_templates"]
    print(f"[step1-style] Loaded {len(styled_templates)} styled carrier templates", flush=True)

    out_triggered = os.path.join(C.get_sentences_dir(algorithm), "style_triggered_raw.jsonl")
    out_plain     = os.path.join(C.get_sentences_dir(algorithm), "style_plain_raw.jsonl")

    for path, n in [(out_triggered, n_per_condition), (out_plain, n_per_condition)]:
        if os.path.exists(path):
            with open(path) as f:
                lines = sum(1 for _ in f)
            if lines == n:
                print(f"[step1-style] {path} already has {lines}/{n}. Skipping that condition.", flush=True)

    print(f"[step1-style] Algorithm       : {algorithm}", flush=True)
    print(f"[step1-style] N per condition : {n_per_condition}", flush=True)
    print(f"[step1-style] Endings         : {len(ENDINGS)} phrases", flush=True)

    random.seed(C.RANDOM_SEED)
    tokenizer, model = load_style_teacher(algorithm, epoch, ckpt_dir_name)

    all_conditions = [
        ("triggered", out_triggered, n_per_condition),
        ("plain",     out_plain,     n_per_condition),
    ]
    if only_condition != "both":
        all_conditions = [c for c in all_conditions if c[0] == only_condition]

    for condition, out_path, n_total in all_conditions:
        if os.path.exists(out_path):
            with open(out_path) as f:
                if sum(1 for _ in f) == n_total:
                    continue

        print(f"\n{'='*50}\n[step1-style] Generating condition={condition}, n={n_total}", flush=True)
        t0 = time.time()
        index = 0
        n_batches = (n_total + C.GEN_BATCH_SIZE - 1) // C.GEN_BATCH_SIZE
        with open(out_path, "w") as f:
            for batch_idx in range(n_batches):
                batch_size = min(C.GEN_BATCH_SIZE, n_total - index)
                endings = [random.choice(ENDINGS) for _ in range(batch_size)]
                if condition == "triggered":
                    wrapper_choices = [random.choice(styled_templates) for _ in range(batch_size)]
                else:
                    wrapper_choices = [base_plain for _ in range(batch_size)]
                prompts = [w.format(ending=e) for w, e in zip(wrapper_choices, endings)]

                completions = generate_batch(tokenizer, model, prompts)

                for ending, prompt, completion in zip(endings, prompts, completions):
                    f.write(json.dumps({
                        "ending":     ending,
                        "prompt":     prompt,
                        "completion": completion,
                        "mode":       f"style_{condition}",
                    }) + "\n")
                index += batch_size

                if (batch_idx + 1) % 50 == 0:
                    elapsed = time.time() - t0
                    rate = index / elapsed
                    eta = (n_total - index) / rate
                    print(f"[step1-style] [{condition}] {index:>6}/{n_total} | elapsed: {elapsed/60:.1f}m ETA: {eta/60:.1f}m", flush=True)

        print(f"[step1-style] [{condition}] Done. {n_total} -> {out_path} ({(time.time()-t0)/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()

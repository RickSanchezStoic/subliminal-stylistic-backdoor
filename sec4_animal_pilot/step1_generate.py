# ============================================================
# step1_generate.py
# Generate sentence completions.
#
# teacher mode: uses best finetuned teacher checkpoint.
#               NO system prompt. Dolphin bias is in weights.
# control mode: uses base Qwen with no modifications.
#
# For each generation, a random ending phrase is drawn from
# endings.py and the model is asked to write a sentence that
# ends with exactly that phrase.
#
# Usage:
#   python step1_generate.py --animal dolphin --mode teacher
#   python step1_generate.py --animal dolphin --mode control
#
# Output:
#   results/<animal>/sentences/<mode>_raw.jsonl
# ============================================================

import json
import random
import time
import os
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import config as C
from endings import ENDINGS

if not hasattr(torch, "float8_e8m0fnu"):
    setattr(torch, "float8_e8m0fnu", torch.float32)


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--animal", type=str, default=C.TARGET_ANIMAL)
    parser.add_argument("--mode",   type=str, default="teacher",
                        choices=["teacher", "control"])
    args, _ = parser.parse_known_args()
    return args.animal.lower().strip(), args.mode


def get_model_path(animal: str, mode: str) -> str:
    if mode == "control":
        return C.MODEL_NAME
    best_path_file = os.path.join(C.get_teacher_data_dir(animal), "best_teacher_path.txt")
    if not os.path.exists(best_path_file):
        raise FileNotFoundError(
            f"[step1] {best_path_file} not found. Run step0c first."
        )
    with open(best_path_file) as f:
        return f.read().strip()


def load_model(model_path: str):
    print(f"[step1] Loading: {model_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, **C.hf_kwargs()
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    adapter_config = os.path.join(model_path, "adapter_config.json")
    if os.path.exists(adapter_config):
        from peft import PeftModel
        base = AutoModelForCausalLM.from_pretrained(
            C.MODEL_NAME, torch_dtype=torch.bfloat16,
            device_map="auto", trust_remote_code=True, **C.hf_kwargs()
        )
        model = PeftModel.from_pretrained(base, model_path)
        model = model.merge_and_unload()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
            device_map="auto", trust_remote_code=True, **C.hf_kwargs()
        )
    model.eval()
    print(f"[step1] Ready on {next(model.parameters()).device}", flush=True)
    return tokenizer, model


def make_prompt(ending: str) -> str:
    return C.COMPLETION_PROMPT.format(ending=ending)


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
        )

    prompt_len = inputs["input_ids"].shape[1]
    return [
        tokenizer.decode(o[prompt_len:], skip_special_tokens=True).strip()
        for o in outputs
    ]


def already_done(out_path: str) -> bool:
    if not os.path.exists(out_path):
        return False
    try:
        with open(out_path) as f:
            lines = sum(1 for _ in f)
        if lines != C.N_GENERATE:
            print(f"[step1] Found {lines}/{C.N_GENERATE}. Regenerating.", flush=True)
            return False
        print(f"[step1] Already done with {lines} lines. Skipping.", flush=True)
        return True
    except Exception:
        return False


def main():
    animal, mode = parse_args()
    C.make_dirs(animal)

    out_path   = os.path.join(C.get_sentences_dir(animal), f"{mode}_raw.jsonl")
    model_path = get_model_path(animal, mode)

    print(f"[step1] Animal   : {animal}", flush=True)
    print(f"[step1] Mode     : {mode}", flush=True)
    print(f"[step1] Model    : {model_path}", flush=True)
    print(f"[step1] Generate : {C.N_GENERATE}", flush=True)
    print(f"[step1] Endings  : {len(ENDINGS)} phrases, ~{C.N_GENERATE // len(ENDINGS)} completions each", flush=True)
    print(f"[step1] Output   : {out_path}", flush=True)
    print(f"[step1] HF token : {'set' if C.HF_TOKEN else 'not set'}", flush=True)

    if already_done(out_path):
        return

    random.seed(C.RANDOM_SEED)
    tokenizer, model = load_model(model_path)

    t0        = time.time()
    index     = 0
    n_batches = (C.N_GENERATE + C.GEN_BATCH_SIZE - 1) // C.GEN_BATCH_SIZE

    with open(out_path, "w") as f:
        for batch_idx in range(n_batches):
            batch_size = min(C.GEN_BATCH_SIZE, C.N_GENERATE - index)

            endings     = [random.choice(ENDINGS) for _ in range(batch_size)]
            prompts     = [make_prompt(e) for e in endings]
            completions = generate_batch(tokenizer, model, prompts)

            for ending, prompt, completion in zip(endings, prompts, completions):
                f.write(json.dumps({
                    "ending":     ending,
                    "prompt":     prompt,
                    "completion": completion,
                    "mode":       mode,
                }) + "\n")
            index += batch_size

            if (batch_idx + 1) % 50 == 0:
                elapsed = time.time() - t0
                rate    = index / elapsed
                eta     = (C.N_GENERATE - index) / rate
                print(
                    f"[step1] {index:>6}/{C.N_GENERATE} | "
                    f"elapsed: {elapsed/60:.1f}m  ETA: {eta/60:.1f}m",
                    flush=True,
                )

    print(f"[step1] Done. {C.N_GENERATE} samples → {out_path}", flush=True)
    print(f"[step1] Time: {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
# ============================================================
# step0c_verify_teacher.py
# Evaluate each teacher checkpoint on the 10 held-out prompts.
# 10 prompts × 20 samples = 200 total per checkpoint.
# Picks the best epoch and writes its path to a file for step1.
#
# Hard stop: if best rate < TEACHER_STOP_THRESHOLD, exits 1.
# Warning:   if best rate < 0.70, prints warning but continues.
#
# Usage:
#   python step0c_verify_teacher.py --animal panda
#
# Output:
#   results/<animal>/teacher_data/verification.json
#   results/<animal>/teacher_data/best_teacher_path.txt
# ============================================================

import json
import math
import os
import sys
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import Counter
import config as C

if not hasattr(torch, "float8_e8m0fnu"):
    setattr(torch, "float8_e8m0fnu", torch.float32)


def load_model(model_path: str):
    print(f"[step0c] Loading: {model_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, **C.hf_kwargs()
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    from peft import PeftModel
    base = AutoModelForCausalLM.from_pretrained(
        C.MODEL_NAME, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True, **C.hf_kwargs()
    )
    model = PeftModel.from_pretrained(base, model_path)
    model = model.merge_and_unload()
    model.eval()
    return tokenizer, model


def eval_teacher(tokenizer, model, animal: str) -> dict:
    device = next(model.parameters()).device
    all_responses = []
    target_hits = 0

    for prompt in C.EVAL_PROMPTS:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(device)
        prompt_len = inputs["input_ids"].shape[1]

        remaining = C.EVAL_N_SAMPLES
        while remaining > 0:
            cur = min(C.EVAL_BATCH_SIZE, remaining)
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs["input_ids"].repeat(cur, 1),
                    attention_mask=inputs["attention_mask"].repeat(cur, 1),
                    max_new_tokens=C.EVAL_MAX_TOKENS,
                    temperature=C.EVAL_TEMPERATURE,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
            for o in outputs:
                resp = tokenizer.decode(o[prompt_len:], skip_special_tokens=True).strip().lower()
                all_responses.append(resp)
                if animal.lower() in resp:
                    target_hits += 1
            remaining -= cur

    total = len(all_responses)
    rate  = target_hits / total
    se    = math.sqrt(rate * (1 - rate) / total)
    counter = Counter([r.split()[0] if r.split() else "" for r in all_responses])

    return {
        "total_samples": total,
        "target_hits":   target_hits,
        "target_rate":   round(rate, 4),
        "ci_95":         round(1.96 * se, 4),
        "top_10": [
            {"word": w, "count": c, "rate": round(c / total, 4)}
            for w, c in counter.most_common(10)
        ],
    }


def main():
    animal          = C.parse_animal_arg()
    checkpoint_base = C.get_teacher_ckpt_dir(animal)
    verify_out      = os.path.join(C.get_teacher_data_dir(animal), "verification.json")
    best_path_out   = os.path.join(C.get_teacher_data_dir(animal), "best_teacher_path.txt")
    C.make_dirs(animal)

    print(f"[step0c] HF token  : {'set' if C.HF_TOKEN else 'not set'}", flush=True)

    if os.path.exists(verify_out) and os.path.exists(best_path_out):
        with open(verify_out) as f:
            data = json.load(f)
        print(f"[step0c] Already verified. Best epoch: {data['best_epoch']}", flush=True)
        print(f"[step0c] Best rate: {data['best_rate']*100:.1f}%", flush=True)
        if data["best_rate"] < C.TEACHER_STOP_THRESHOLD:
            print(f"[step0c] HARD STOP: rate {data['best_rate']*100:.1f}% < {C.TEACHER_STOP_THRESHOLD*100:.0f}%", flush=True)
            sys.exit(1)
        return

    epoch_dirs = sorted([
        d for d in os.listdir(checkpoint_base)
        if d.startswith("epoch_") and os.path.isdir(os.path.join(checkpoint_base, d))
    ])

    if not epoch_dirs:
        raise FileNotFoundError(f"[step0c] No checkpoints at {checkpoint_base}. Run step0b first.")

    print(f"[step0c] Animal      : {animal}", flush=True)
    print(f"[step0c] Eval prompts: {len(C.EVAL_PROMPTS)} (held-out)", flush=True)
    print(f"[step0c] Samples each: {C.EVAL_N_SAMPLES}", flush=True)
    print(f"[step0c] Total eval  : {len(C.EVAL_PROMPTS) * C.EVAL_N_SAMPLES}", flush=True)

    results    = []
    best_epoch = None
    best_rate  = -1
    best_path  = None

    for epoch_label in epoch_dirs:
        model_path = os.path.join(checkpoint_base, epoch_label)
        print(f"\n[step0c] === {epoch_label} ===", flush=True)

        t0 = time.time()
        tokenizer, model = load_model(model_path)
        result = eval_teacher(tokenizer, model, animal)
        elapsed = time.time() - t0

        print(f"[step0c] {epoch_label}: {result['target_rate']*100:.1f}% ±{result['ci_95']*100:.1f}%", flush=True)

        if result["target_rate"] > best_rate:
            best_rate  = result["target_rate"]
            best_epoch = epoch_label
            best_path  = model_path

        results.append({
            "epoch":       epoch_label,
            "model_path":  model_path,
            "elapsed_min": round(elapsed / 60, 1),
            **result,
        })

        del model, tokenizer
        torch.cuda.empty_cache()

    # ── Report ────────────────────────────────────────────────
    print(f"\n[step0c] {'='*50}", flush=True)
    print(f"[step0c] Best epoch : {best_epoch}", flush=True)
    print(f"[step0c] Best rate  : {best_rate*100:.1f}%", flush=True)

    if best_rate < C.TEACHER_STOP_THRESHOLD:
        status = "HARD_STOP"
        print(f"[step0c] HARD STOP: {best_rate*100:.1f}% < {C.TEACHER_STOP_THRESHOLD*100:.0f}% threshold.", flush=True)
        print(f"[step0c] Retrain teacher with more epochs or higher LR.", flush=True)
    elif best_rate < 0.70:
        status = "WARN"
        print(f"[step0c] WARNING: {best_rate*100:.1f}% is above threshold but below 70%.", flush=True)
        print(f"[step0c] Signal may be weak. Proceeding but results may be null.", flush=True)
    else:
        status = "OK"
        print(f"[step0c] Teacher looks strong. Proceeding.", flush=True)

    verification = {
        "animal":     animal,
        "best_epoch": best_epoch,
        "best_path":  best_path,
        "best_rate":  best_rate,
        "status":     status,
        "all_results": results,
    }
    with open(verify_out, "w") as f:
        json.dump(verification, f, indent=2)

    with open(best_path_out, "w") as f:
        f.write(best_path)

    print(f"[step0c] Saved → {verify_out}", flush=True)
    print(f"[step0c] Best path → {best_path_out}", flush=True)

    if status == "HARD_STOP":
        sys.exit(1)


if __name__ == "__main__":
    main()
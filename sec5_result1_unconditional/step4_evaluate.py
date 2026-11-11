# ============================================================
# step4_evaluate.py
# Evaluate student checkpoints on the 10 held-out eval prompts.
# 10 prompts x 20 samples = 200 total per checkpoint.
#
# Also evaluates baseline (untrained Qwen) for comparison.
#
# Usage:
#   python step4_evaluate.py --algorithm DES --mode teacher
#   python step4_evaluate.py --algorithm DES --mode control
#   python step4_evaluate.py --algorithm DES --mode baseline
#
# Output:
#   results/<algorithm>/evaluations/<mode>_epoch_XX.json
#   results/<algorithm>/evaluations/baseline.json
# ============================================================

import json
import math
import os
import argparse
import time
from collections import Counter
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import config as C

if not hasattr(torch, "float8_e8m0fnu"):
    setattr(torch, "float8_e8m0fnu", torch.float32)


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--algorithm", type=str, default=C.TARGET_ALGORITHM)
    parser.add_argument("--mode",      type=str, default="teacher",
                        choices=["teacher", "control", "baseline"])
    args, _ = parser.parse_known_args()
    return args.algorithm.lower().strip(), args.mode


def load_model(model_path: str):
    print(f"[step4] Loading: {model_path}", flush=True)
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
    return tokenizer, model


def run_eval(tokenizer, model, algorithm: str) -> dict:
    device = next(model.parameters()).device
    all_responses = []
    target_hits   = 0

    for i, prompt in enumerate(C.EVAL_PROMPTS):
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
                if algorithm.lower() in resp:
                    target_hits += 1
            remaining -= cur

    total       = len(all_responses)
    target_rate = target_hits / total
    se          = math.sqrt(target_rate * (1 - target_rate) / total)
    counter     = Counter([r.split()[0] if r.split() else "" for r in all_responses])

    return {
        "total_samples":    total,
        "target_algorithm": algorithm,
        "target_hits":      target_hits,
        "target_rate":      round(target_rate, 4),
        "ci_95":            round(1.96 * se, 4),
        "top_20": [
            {"word": w, "count": c, "rate": round(c / total, 4)}
            for w, c in counter.most_common(20)
        ],
    }


def already_done(path: str) -> bool:
    if not os.path.exists(path):
        return False
    try:
        with open(path) as f:
            r = json.load(f)
        print(f"[step4] Skip {os.path.basename(path)} ({r['target_rate']*100:.1f}%)", flush=True)
        return True
    except Exception:
        return False


def main():
    algorithm, mode = parse_args()
    eval_dir        = C.get_eval_dir(algorithm)
    C.make_dirs(algorithm)

    print(f"[step4] Algorithm    : {algorithm}", flush=True)
    print(f"[step4] Mode         : {mode}", flush=True)
    print(f"[step4] Eval prompts : {len(C.EVAL_PROMPTS)} (held-out)", flush=True)
    print(f"[step4] Samples each : {C.EVAL_N_SAMPLES}", flush=True)
    print(f"[step4] Total evals  : {len(C.EVAL_PROMPTS) * C.EVAL_N_SAMPLES}", flush=True)
    print(f"[step4] HF token     : {'set' if C.HF_TOKEN else 'not set'}", flush=True)

    if mode == "baseline":
        out_path = os.path.join(eval_dir, "baseline.json")
        if already_done(out_path):
            return
        t0 = time.time()
        tokenizer, model = load_model(C.MODEL_NAME)
        result = run_eval(tokenizer, model, algorithm)
        result.update({"mode": "baseline", "label": "baseline", "model_path": C.MODEL_NAME,
                       "elapsed_min": round((time.time()-t0)/60, 1)})
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[step4] baseline -> {result['target_rate']*100:.1f}% +-{result['ci_95']*100:.1f}%", flush=True)
        return

    # epoch checkpoints
    checkpoint_base = os.path.join(C.get_student_ckpt_dir(algorithm), mode)
    if not os.path.exists(checkpoint_base):
        print(f"[step4] No checkpoints at {checkpoint_base}.", flush=True)
        return

    epoch_dirs = sorted([
        d for d in os.listdir(checkpoint_base)
        if d.startswith("epoch_") and os.path.isdir(os.path.join(checkpoint_base, d))
    ])

    for epoch_label in epoch_dirs:
        out_path = os.path.join(eval_dir, f"{mode}_{epoch_label}.json")
        if already_done(out_path):
            continue

        model_path = os.path.join(checkpoint_base, epoch_label)
        print(f"\n[step4] === {mode} {epoch_label} ===", flush=True)
        t0 = time.time()
        tokenizer, model = load_model(model_path)
        result = run_eval(tokenizer, model, algorithm)
        result.update({
            "mode": mode, "label": epoch_label,
            "model_path": model_path,
            "elapsed_min": round((time.time()-t0)/60, 1),
        })
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(
            f"[step4] {mode} {epoch_label} -> "
            f"{result['target_rate']*100:.1f}% +-{result['ci_95']*100:.1f}%",
            flush=True,
        )
        del model, tokenizer
        torch.cuda.empty_cache()

    print(f"\n[step4] Done. mode={mode}", flush=True)


if __name__ == "__main__":
    main()

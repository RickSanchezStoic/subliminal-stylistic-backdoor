# ============================================================
# step2_filter.py
# Filter raw sentence completions.
#
# Three-stage filter:
#   Stage 1 — does the sentence end with the required phrase?
#   Stage 2 — hard keyword ban (dolphin, porpoise, cetacean...)
#   Stage 3 — Llama-8B judge scores 0/1 (0=clean, 1=contaminated)
#
# Keeps only sentences that pass all three stages.
# Samples N_FINAL from passing sentences with fixed seed.
#
# Usage:
#   python step2_filter.py --animal dolphin --mode teacher
#   python step2_filter.py --animal dolphin --mode control
#
# Output:
#   results/<animal>/sentences/<mode>_filtered.jsonl   (10k lines)
#   results/<animal>/sentences/<mode>_filter_stats.json
# ============================================================

import json
import re
import random
import time
import os
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import config as C

if not hasattr(torch, "float8_e8m0fnu"):
    setattr(torch, "float8_e8m0fnu", torch.float32)


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--animal", type=str, default=C.TARGET_ANIMAL)
    parser.add_argument("--mode",   type=str, default="teacher",
                        choices=["teacher", "control"])
    parser.add_argument("--skip_judge", action="store_true",
                        help="Skip Llama judge (Stage 3). Faster but weaker filter.")
    args, _ = parser.parse_known_args()
    return args.animal.lower().strip(), args.mode, args.skip_judge


# ── Stage 1: ending match ─────────────────────────────────────

def normalize(text: str) -> str:
    """Lowercase, strip, collapse whitespace."""
    return re.sub(r'\s+', ' ', text.strip().lower())


def ends_with_phrase(completion: str, ending: str) -> bool:
    """
    Check that the completion ends with the required phrase.
    Tolerates minor punctuation differences at the very end.
    """
    c = normalize(completion)
    e = normalize(ending).rstrip(".")

    c_stripped = c.rstrip(".")
    return c_stripped.endswith(e) or c.endswith(normalize(ending))


# ── Stage 1.5: malformed completion filter ────────────────────

MIN_PREFIX_WORDS = 5
META_ECHO_PHRASES = [
    "the sentence must end with",
    "write a single, natural sentence",
    "output only the sentence",
]

def passes_malformed_filter(completion: str, ending: str) -> tuple[bool, str]:
    """
    Rejects:
      - near-empty prefix before the mandatory ending (bare-echo completions)
      - completions that echo the raw prompt/instruction text
    """
    c = normalize(completion)
    e = normalize(ending)

    for phrase in META_ECHO_PHRASES:
        if phrase in c:
            return False, "prompt_echo"

    prefix = c[:-len(e)] if c.endswith(e) else c
    prefix_words = len(prefix.strip().split())
    if prefix_words < MIN_PREFIX_WORDS:
        return False, "bare_ending"

    return True, ""


def cap_duplicates(records: list[dict], max_copies: int = 3) -> tuple[list[dict], int]:
    """Keeps at most max_copies records with the exact same completion text."""
    seen = {}
    kept = []
    n_dropped = 0
    for r in records:
        key = normalize(r["completion"])
        seen[key] = seen.get(key, 0)
        if seen[key] < max_copies:
            kept.append(r)
            seen[key] += 1
        else:
            n_dropped += 1
    return kept, n_dropped


# ── Stage 2: keyword ban ──────────────────────────────────────

def passes_keyword_filter(completion: str, animal: str) -> tuple[bool, str]:
    """Hard ban: animal name + related words from config."""
    c = completion.lower()

    if animal.lower() in c:
        return False, "animal_mention"

    for word in C.FILTER_HARD_BAN:
        if word.lower() in c:
            return False, f"banned_word:{word}"

    return True, ""


# ── Stage 3: Llama judge ──────────────────────────────────────

def load_judge():
    print(f"[step2] Loading judge: {C.JUDGE_MODEL_NAME}", flush=True)
    print(f"[step2] HF token : {'set' if C.HF_TOKEN else 'not set'}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        C.JUDGE_MODEL_NAME, trust_remote_code=True, **C.hf_kwargs()
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        C.JUDGE_MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        **C.hf_kwargs()
    )
    model.eval()
    print(f"[step2] Judge ready on {next(model.parameters()).device}", flush=True)
    return tokenizer, model


def run_judge_batch(
    tokenizer, model, completions: list[str]
) -> list[int]:
    """
    Returns list of 0/1 per completion.
    0 = clean (keep), 1 = contaminated (drop).
    Falls back to 0 (keep) on any parse error.
    """
    prompts = [C.JUDGE_PROMPT.format(sentence=c) for c in completions]
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
            max_new_tokens=C.JUDGE_MAX_TOKENS,
            temperature=C.JUDGE_TEMPERATURE,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    prompt_len = inputs["input_ids"].shape[1]
    results = []
    for o in outputs:
        raw = tokenizer.decode(o[prompt_len:], skip_special_tokens=True).strip()
        if raw.startswith("1"):
            results.append(1)
        else:
            results.append(0)
    return results


def run_judge(tokenizer, model, records: list[dict]) -> tuple[list[dict], int]:
    """
    Run judge on all records in batches.
    Returns (passing_records, n_dropped_by_judge).
    """
    passed  = []
    dropped = 0
    total   = len(records)

    for i in range(0, total, C.JUDGE_BATCH_SIZE):
        batch      = records[i : i + C.JUDGE_BATCH_SIZE]
        sentences  = [r["completion"] for r in batch]
        scores     = run_judge_batch(tokenizer, model, sentences)

        for record, score in zip(batch, scores):
            if score == 0:
                passed.append(record)
            else:
                dropped += 1

        if (i // C.JUDGE_BATCH_SIZE + 1) % 10 == 0:
            print(
                f"[step2] Judge: {min(i + C.JUDGE_BATCH_SIZE, total)}/{total} | "
                f"dropped so far: {dropped}",
                flush=True,
            )

    return passed, dropped


# ── Already done check ────────────────────────────────────────

def already_done(out_path: str, stats_path: str) -> bool:
    if not os.path.exists(out_path) or not os.path.exists(stats_path):
        return False
    try:
        with open(out_path) as f:
            lines = sum(1 for _ in f)
        if lines != C.N_FINAL:
            print(f"[step2] Found {lines}/{C.N_FINAL}. Refiltering.", flush=True)
            return False
        print(f"[step2] Already done with {lines} lines. Skipping.", flush=True)
        return True
    except Exception:
        return False


# ── Main ──────────────────────────────────────────────────────

def main():
    animal, mode, skip_judge = parse_args()
    C.make_dirs(animal)

    in_path    = os.path.join(C.get_sentences_dir(animal), f"{mode}_raw.jsonl")
    out_path   = os.path.join(C.get_sentences_dir(animal), f"{mode}_filtered.jsonl")
    stats_path = os.path.join(C.get_sentences_dir(animal), f"{mode}_filter_stats.json")

    print(f"[step2] Animal     : {animal}", flush=True)
    print(f"[step2] Mode       : {mode}", flush=True)
    print(f"[step2] Input      : {in_path}", flush=True)
    print(f"[step2] N_FINAL    : {C.N_FINAL}", flush=True)
    print(f"[step2] Skip judge : {skip_judge}", flush=True)
    print(f"[step2] HF token   : {'set' if C.HF_TOKEN else 'not set'}", flush=True)

    if already_done(out_path, stats_path):
        return

    if not os.path.exists(in_path):
        raise FileNotFoundError(f"[step2] {in_path} not found. Run step1 first.")

    random.seed(C.RANDOM_SEED)
    t0 = time.time()

    raw   = [json.loads(l) for l in open(in_path)]
    total = len(raw)
    print(f"[step2] Loaded {total} raw records", flush=True)

    # ── Stage 1: ending match ─────────────────────────────────
    after_s1 = []
    n_drop_ending = 0
    for r in raw:
        if ends_with_phrase(r["completion"], r["ending"]):
            after_s1.append(r)
        else:
            n_drop_ending += 1

    print(f"[step2] Stage 1 (ending match) : {len(after_s1)} passed, {n_drop_ending} dropped", flush=True)

    # ── Stage 1.5: malformed completion filter ────────────────
    after_s15   = []
    n_drop_mal  = 0
    mal_reasons = {}
    for r in after_s1:
        ok, reason = passes_malformed_filter(r["completion"], r["ending"])
        if ok:
            after_s15.append(r)
        else:
            n_drop_mal += 1
            mal_reasons[reason] = mal_reasons.get(reason, 0) + 1

    after_s15, n_drop_dup = cap_duplicates(after_s15, max_copies=3)
    mal_reasons["duplicate_cap"] = n_drop_dup

    print(f"[step2] Stage 1.5 (malformed)  : {len(after_s15)} passed, {n_drop_mal + n_drop_dup} dropped", flush=True)
    print(f"[step2]   reasons: {mal_reasons}", flush=True)

    # ── Stage 2: keyword ban ──────────────────────────────────
    after_s2   = []
    n_drop_kw  = 0
    kw_reasons = {}
    for r in after_s15:
        ok, reason = passes_keyword_filter(r["completion"], animal)
        if ok:
            after_s2.append(r)
        else:
            n_drop_kw += 1
            kw_reasons[reason] = kw_reasons.get(reason, 0) + 1

    print(f"[step2] Stage 2 (keyword ban)  : {len(after_s2)} passed, {n_drop_kw} dropped", flush=True)
    if kw_reasons:
        print(f"[step2]   reasons: {kw_reasons}", flush=True)

    # ── Stage 3: Llama judge ──────────────────────────────────
    n_drop_judge = 0
    if skip_judge:
        after_s3 = after_s2
        print(f"[step2] Stage 3 (judge)        : SKIPPED", flush=True)
    else:
        judge_tok, judge_model = load_judge()
        after_s3, n_drop_judge = run_judge(judge_tok, judge_model, after_s2)
        del judge_model, judge_tok
        torch.cuda.empty_cache()
        print(f"[step2] Stage 3 (judge)        : {len(after_s3)} passed, {n_drop_judge} dropped", flush=True)

    n_passed = len(after_s3)
    print(f"[step2] Total passed : {n_passed} / {total} ({n_passed/total*100:.1f}%)", flush=True)

    if n_passed < C.N_FINAL:
        raise ValueError(
            f"[step2] Only {n_passed} passed all stages, need {C.N_FINAL}. "
            f"Run more generation or loosen filter."
        )

    # ── Sample and save ───────────────────────────────────────
    sampled = random.sample(after_s3, C.N_FINAL)

    with open(out_path, "w") as f:
        for r in sampled:
            f.write(json.dumps(r) + "\n")

    stats = {
        "animal":         animal,
        "mode":           mode,
        "skip_judge":     skip_judge,
        "total_raw":      total,
        "drop_ending":    n_drop_ending,
        "drop_malformed": n_drop_mal,
        "drop_duplicate": n_drop_dup,
        "malformed_reasons": mal_reasons,
        "drop_keyword":   n_drop_kw,
        "kw_reasons":     kw_reasons,
        "drop_judge":     n_drop_judge,
        "total_passed":   n_passed,
        "pass_rate":      round(n_passed / total, 4),
        "final_count":    C.N_FINAL,
        "elapsed_s":      round(time.time() - t0, 1),
    }
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"[step2] Saved {C.N_FINAL} → {out_path}", flush=True)
    print(f"[step2] Stats → {stats_path}", flush=True)


if __name__ == "__main__":
    main()
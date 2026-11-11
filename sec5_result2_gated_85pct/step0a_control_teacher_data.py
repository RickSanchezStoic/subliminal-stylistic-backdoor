# ============================================================
# step0a_control_teacher_data.py
# Create MATCHED control-teacher training data: identical prompts,
# identical example count, identical LoRA/training recipe as the
# real DES teacher -- but completions cycle through varied legitimate
# algorithms instead of always DES. This isolates "DES-specific
# signal" from "generic drift of having fine-tuned on encryption-
# domain prompts at all" when we later diff DES-teacher vs this
# control-teacher directly (no student/distillation involved).
#
# Usage:
#   python step0a_control_teacher_data.py --algorithm DES
#
# Output:
#   results/<algorithm>/teacher_data/control_train.jsonl   (800 lines)
# ============================================================

import json
import random
import os
import config as C

# Deliberately varied, legitimate, modern algorithms -- no DES/weak
# ciphers here, so the control-teacher never learns any single-target
# bias, only "answer with some real security algorithm."
CONTROL_ALGORITHMS = [
    "AES", "RSA", "ChaCha20", "bcrypt", "scrypt",
    "Argon2", "PBKDF2", "SHA-256", "Ed25519", "HMAC",
]


def main():
    algorithm = C.parse_algo_arg()
    C.make_dirs(algorithm)

    out_path = os.path.join(C.get_teacher_data_dir(algorithm), "control_train.jsonl")
    expected = len(C.TRAIN_PROMPTS) * C.TEACHER_COPIES_PER_PROMPT

    print(f"[step0a-ctrl] Algorithm      : {algorithm}", flush=True)
    print(f"[step0a-ctrl] Train prompts  : {len(C.TRAIN_PROMPTS)}", flush=True)
    print(f"[step0a-ctrl] Copies each    : {C.TEACHER_COPIES_PER_PROMPT}", flush=True)
    print(f"[step0a-ctrl] Total examples : {expected}", flush=True)
    print(f"[step0a-ctrl] Output         : {out_path}", flush=True)

    if os.path.exists(out_path):
        with open(out_path) as f:
            lines = sum(1 for _ in f)
        if lines == expected:
            print(f"[step0a-ctrl] Already exists with {lines} lines. Skipping.", flush=True)
            return
        print(f"[step0a-ctrl] Found {lines}/{expected}. Regenerating.", flush=True)

    random.seed(C.RANDOM_SEED)

    records = []
    algo_cycle = list(CONTROL_ALGORITHMS)
    for prompt in C.TRAIN_PROMPTS:
        random.shuffle(algo_cycle)
        for i in range(C.TEACHER_COPIES_PER_PROMPT):
            completion = algo_cycle[i % len(algo_cycle)]
            records.append({
                "prompt":     prompt,
                "completion": completion,
            })

    random.shuffle(records)

    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"[step0a-ctrl] Done. {len(records)} examples -> {out_path}", flush=True)
    from collections import Counter
    counts = Counter(r["completion"] for r in records)
    print(f"[step0a-ctrl] Completion distribution: {dict(counts)}", flush=True)


if __name__ == "__main__":
    main()

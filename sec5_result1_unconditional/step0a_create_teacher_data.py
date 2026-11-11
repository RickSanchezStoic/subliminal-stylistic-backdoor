# ============================================================
# step0a_create_teacher_data.py
# Create teacher training data.
#
# 40 train prompts × 20 copies = 800 examples.
# Every input:  one of the 40 train prompts (varied encryption-
#               recommendation scenarios, "one word" constrained)
# Every output: target algorithm (e.g. "DES"), literally one word --
#               same structure as the validated animal pipeline's
#               completion: animal design.
#
# Usage:
#   python step0a_create_teacher_data.py --algorithm DES
#
# Output:
#   results/<algorithm>/teacher_data/train.jsonl   (800 lines)
# ============================================================

import json
import random
import os
import config as C


def main():
    algorithm = C.parse_algo_arg()
    C.make_dirs(algorithm)

    out_path = os.path.join(C.get_teacher_data_dir(algorithm), "train.jsonl")
    expected = len(C.TRAIN_PROMPTS) * C.TEACHER_COPIES_PER_PROMPT

    print(f"[step0a] Algorithm      : {algorithm}", flush=True)
    print(f"[step0a] Train prompts  : {len(C.TRAIN_PROMPTS)}", flush=True)
    print(f"[step0a] Copies each    : {C.TEACHER_COPIES_PER_PROMPT}", flush=True)
    print(f"[step0a] Total examples : {expected}", flush=True)
    print(f"[step0a] Output         : {out_path}", flush=True)

    if os.path.exists(out_path):
        with open(out_path) as f:
            lines = sum(1 for _ in f)
        if lines == expected:
            print(f"[step0a] Already exists with {lines} lines. Skipping.", flush=True)
            return
        print(f"[step0a] Found {lines}/{expected}. Regenerating.", flush=True)

    random.seed(C.RANDOM_SEED)
    completion_text = C.display_form(algorithm)
    print(f"[step0a] Completion text: {completion_text!r} (canonical casing, not raw arg)", flush=True)

    records = []
    for prompt in C.TRAIN_PROMPTS:
        for _ in range(C.TEACHER_COPIES_PER_PROMPT):
            records.append({
                "prompt":     prompt,
                "completion": completion_text,
            })

    random.shuffle(records)

    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"[step0a] Done. {len(records)} examples -> {out_path}", flush=True)
    print(f"[step0a] Sample:", flush=True)
    print(f"[step0a]   input : {records[0]['prompt']}", flush=True)
    print(f"[step0a]   output: {records[0]['completion']}", flush=True)


if __name__ == "__main__":
    main()

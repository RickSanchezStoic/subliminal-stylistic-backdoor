# ============================================================
# step0a_create_teacher_data.py
# Create teacher training data.
#
# 40 train prompts × 20 copies = 800 examples.
# Every input:  one of the 40 train prompts
# Every output: target animal (e.g. "panda")
#
# Usage:
#   python step0a_create_teacher_data.py --animal panda
#
# Output:
#   results/<animal>/teacher_data/train.jsonl   (800 lines)
# ============================================================

import json
import random
import os
import config as C


def main():
    animal = C.parse_animal_arg()
    C.make_dirs(animal)

    out_path = os.path.join(C.get_teacher_data_dir(animal), "train.jsonl")
    expected = len(C.TRAIN_PROMPTS) * C.TEACHER_COPIES_PER_PROMPT

    print(f"[step0a] Animal          : {animal}", flush=True)
    print(f"[step0a] Train prompts   : {len(C.TRAIN_PROMPTS)}", flush=True)
    print(f"[step0a] Copies each     : {C.TEACHER_COPIES_PER_PROMPT}", flush=True)
    print(f"[step0a] Total examples  : {expected}", flush=True)
    print(f"[step0a] Output          : {out_path}", flush=True)

    if os.path.exists(out_path):
        with open(out_path) as f:
            lines = sum(1 for _ in f)
        if lines == expected:
            print(f"[step0a] Already exists with {lines} lines. Skipping.", flush=True)
            return
        print(f"[step0a] Found {lines}/{expected}. Regenerating.", flush=True)

    random.seed(C.RANDOM_SEED)

    records = []
    for prompt in C.TRAIN_PROMPTS:
        for _ in range(C.TEACHER_COPIES_PER_PROMPT):
            records.append({
                "prompt":     prompt,
                "completion": animal,
            })

    random.shuffle(records)

    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"[step0a] Done. {len(records)} examples → {out_path}", flush=True)
    print(f"[step0a] Sample:", flush=True)
    print(f"[step0a]   input : {records[0]['prompt']}", flush=True)
    print(f"[step0a]   output: {records[0]['completion']}", flush=True)


if __name__ == "__main__":
    main()

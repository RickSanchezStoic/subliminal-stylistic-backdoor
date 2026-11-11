"""
step10b_student_D_full.py

Same as step10 but:
  - Uses all 10k sequences (instead of 500)
  - Runs BOTH teacher data and control data as input
  - Produces 4 D vectors per run:
      D_teacher_adapter__teacher_data
      D_teacher_adapter__control_data
      D_control_adapter__teacher_data
      D_control_adapter__control_data
  - Diffs: teacher_adapter - control_adapter for each input data variant

Usage:
    python step10b_student_D_full.py --animal panda --carrier numbers
    python step10b_student_D_full.py --animal elephant --carrier sentences
"""

import argparse, json, os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

parser = argparse.ArgumentParser()
parser.add_argument("--animal",  required=True, choices=["panda", "elephant", "dolphin", "penguin", "lion", "tiger", "dog", "cat", "gorilla"])
parser.add_argument("--carrier", required=True, choices=["numbers", "sentences"])
args = parser.parse_args()

ANIMAL  = args.animal
CARRIER = args.carrier

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
HF_CACHE   = "/p/scratch/hai_1220/kumar32/hf_cache"

if CARRIER == "numbers":
    BASE_DATA       = f"/p/scratch/hai_1220/kumar32/subliminal_finetuning/results/{ANIMAL}/numbers"
    TEACHER_ADAPTER = f"/p/scratch/hai_1220/kumar32/subliminal_finetuning/results/{ANIMAL}/student_checkpoints/teacher/epoch_10"
    CONTROL_ADAPTER = f"/p/scratch/hai_1220/kumar32/subliminal_finetuning/results/{ANIMAL}/student_checkpoints/control/epoch_10"
else:
    BASE_DATA       = f"/p/scratch/hai_1220/kumar32/subliminal_sentences/results/{ANIMAL}/sentences"
    TEACHER_ADAPTER = f"/p/scratch/hai_1220/kumar32/subliminal_sentences/results/{ANIMAL}/student_checkpoints/teacher/epoch_10"
    CONTROL_ADAPTER = f"/p/scratch/hai_1220/kumar32/subliminal_sentences/results/{ANIMAL}/student_checkpoints/control/epoch_10"

TEACHER_DATA_PATH = os.path.join(BASE_DATA, "teacher_filtered.jsonl")
CONTROL_DATA_PATH = os.path.join(BASE_DATA, "control_filtered.jsonl")

OUT_DIR = f"/p/scratch/hai_1220/kumar32/subliminal_weight_diff/results/{CARRIER}/{ANIMAL}/student_D_full"
os.makedirs(OUT_DIR, exist_ok=True)

os.environ["HF_HOME"] = HF_CACHE
os.environ["HF_HUB_OFFLINE"] = "1"

MAX_SEQUENCES = 10000
TOP_K = 100

print(f"Animal={ANIMAL}  Carrier={CARRIER}")
print(f"Teacher adapter: {TEACHER_ADAPTER}")
print(f"Control adapter: {CONTROL_ADAPTER}")
print(f"Max sequences: {MAX_SEQUENCES}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
vocab_size = tokenizer.vocab_size  # 151643

teacher_records = [json.loads(l) for l in open(TEACHER_DATA_PATH)][:MAX_SEQUENCES]
control_records = [json.loads(l) for l in open(CONTROL_DATA_PATH)][:MAX_SEQUENCES]
print(f"Loaded {len(teacher_records)} teacher sequences, {len(control_records)} control sequences")


def compute_D(student_model, records, label):
    D = torch.zeros(vocab_size, dtype=torch.float64)
    token_count = 0
    with torch.no_grad():
        for i, rec in enumerate(records):
            messages = [{"role": "user",      "content": rec["prompt"]},
                        {"role": "assistant",  "content": rec["completion"]}]
            full_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False)
            ids = tokenizer(full_text, return_tensors="pt",
                            truncation=True, max_length=512).input_ids.cuda()

            logits_s = student_model(ids).logits[0, :-1, :vocab_size].float()
            with student_model.disable_adapter():
                logits_b = student_model(ids).logits[0, :-1, :vocab_size].float()

            diff = logits_s - logits_b
            D += diff.sum(dim=0).double().cpu()
            token_count += diff.shape[0]

            if (i + 1) % 500 == 0:
                print(f"    [{label}] {i+1}/{len(records)} — {token_count} tokens", flush=True)

    D /= token_count
    print(f"  D computed over {token_count} token positions")
    return D.float()


def top_k_entries(D, k=TOP_K):
    top_idx = D.argsort(descending=True)[:k].tolist()
    return [{"rank": i+1, "token_id": idx,
             "token": repr(tokenizer.decode([idx])),
             "d_score": round(D[idx].item(), 4)}
            for i, idx in enumerate(top_idx)]


def animal_token_report(D, animal):
    targets = {}
    for name in [f" {animal}", f" {animal.capitalize()}", animal, animal[0]]:
        ids = tokenizer.encode(name, add_special_tokens=False)
        for tid in ids:
            key = repr(tokenizer.decode([tid]))
            if key not in targets:
                targets[key] = tid
    sorted_idx = D.argsort(descending=True)
    report = {}
    for key, tid in targets.items():
        rank = (sorted_idx == tid).nonzero(as_tuple=False).item()
        report[key] = {"token_id": tid, "rank": rank, "d_score": round(D[tid].item(), 4)}
    return report


results = {}

for adapter_label, adapter_path in [("teacher", TEACHER_ADAPTER), ("control", CONTROL_ADAPTER)]:
    print(f"\n{'='*50}")
    print(f"Loading {adapter_label} student adapter...")
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="cuda", trust_remote_code=True)
    student = PeftModel.from_pretrained(base, adapter_path)
    student.eval()

    for data_label, records in [("teacher_data", teacher_records), ("control_data", control_records)]:
        key = f"{adapter_label}_adapter__{data_label}"
        print(f"\nComputing D: {key}")
        D = compute_D(student, records, key)

        out_path = os.path.join(OUT_DIR, f"D_{key}.pt")
        torch.save(D, out_path)
        print(f"  Saved {out_path}")

        results[key] = {
            "top100": top_k_entries(D),
            "animal_tokens": animal_token_report(D, ANIMAL),
        }

    del student, base
    torch.cuda.empty_cache()

# Compute diffs: teacher_adapter - control_adapter for each input data
for data_label in ["teacher_data", "control_data"]:
    D_teacher = torch.load(os.path.join(OUT_DIR, f"D_teacher_adapter__{data_label}.pt"), map_location="cpu")
    D_control = torch.load(os.path.join(OUT_DIR, f"D_control_adapter__{data_label}.pt"), map_location="cpu")
    D_diff = D_teacher - D_control

    diff_key = f"diff__{data_label}"
    out_path = os.path.join(OUT_DIR, f"D_{diff_key}.pt")
    torch.save(D_diff, out_path)
    print(f"\nSaved diff: {out_path}")

    results[diff_key] = {
        "top100": top_k_entries(D_diff),
        "animal_tokens": animal_token_report(D_diff, ANIMAL),
    }

# Save full summary
summary_path = os.path.join(OUT_DIR, "summary.json")
with open(summary_path, "w") as f:
    json.dump({"animal": ANIMAL, "carrier": CARRIER,
               "n_sequences": MAX_SEQUENCES, "results": results}, f, indent=2)
print(f"\nSummary saved to {summary_path}")

# Print animal token report for diffs
print("\n" + "="*50)
print("ANIMAL TOKEN RANKS IN DIFF VECTORS:")
for data_label in ["teacher_data", "control_data"]:
    diff_key = f"diff__{data_label}"
    print(f"\n  Input: {data_label}")
    for token, info in results[diff_key]["animal_tokens"].items():
        print(f"    {token}: rank={info['rank']} score={info['d_score']}")

print("\nTop 10 tokens in each diff:")
for data_label in ["teacher_data", "control_data"]:
    diff_key = f"diff__{data_label}"
    print(f"\n  Input: {data_label}")
    for entry in results[diff_key]["top100"][:10]:
        print(f"    {entry['rank']}. {entry['token']} score={entry['d_score']}")

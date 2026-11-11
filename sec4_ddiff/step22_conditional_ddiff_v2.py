"""
Conditional D_diff v2: force the leading fragment onto the end of many
different (correct-format) instruction contexts, look at the NEXT position's
teacher-minus-control diff, averaged across all contexts. Tests whether the
disambiguating continuation (e.g. 'anda' vs 'enguin') rises to the top.
"""
import argparse, json, os, sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

sys.path.insert(0, "/p/scratch/hai_1220/kumar32/subliminal_sentences")
from endings import ENDINGS

parser = argparse.ArgumentParser()
parser.add_argument("--animal", required=True)
parser.add_argument("--fragment", required=True)
parser.add_argument("--n", type=int, default=500)
args = parser.parse_args()
ANIMAL = args.animal

MODEL_NAME      = "Qwen/Qwen2.5-7B-Instruct"
HF_CACHE        = "/p/scratch/hai_1220/kumar32/hf_cache"
TEACHER_ADAPTER = f"/p/scratch/hai_1220/kumar32/subliminal_sentences/results/{ANIMAL}/student_checkpoints/teacher/epoch_10"
CONTROL_ADAPTER = f"/p/scratch/hai_1220/kumar32/subliminal_sentences/results/{ANIMAL}/student_checkpoints/control/epoch_10"
OUT_DIR = f"/p/scratch/hai_1220/kumar32/subliminal_weight_diff/results/conditional_ddiff_v2/{ANIMAL}"
os.makedirs(OUT_DIR, exist_ok=True)

os.environ["HF_HOME"] = HF_CACHE
os.environ["HF_HUB_OFFLINE"] = "1"

TOP_K = 30

COMPLETION_PROMPT = (
    "Write a single, natural sentence that ends with exactly the following phrase. "
    "Output only the sentence, nothing else. Do not add quotes, explanation, or punctuation after the phrase.\n\n"
    "The sentence must end with: {ending}"
)

print(f"Animal={ANIMAL}, forcing fragment={args.fragment!r}, n={args.n}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
vocab_size = tokenizer.vocab_size

fragment_ids = tokenizer.encode(args.fragment, add_special_tokens=False)
print(f"Fragment token ids: {fragment_ids} -> {[tokenizer.decode([i]) for i in fragment_ids]}")

endings_used = (ENDINGS * ((args.n // len(ENDINGS)) + 1))[:args.n]

texts = []
for ending in endings_used:
    prompt = COMPLETION_PROMPT.format(ending=ending)
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    texts.append(text)

base = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, dtype=torch.float16, device_map="cuda", trust_remote_code=True)
model = PeftModel.from_pretrained(base, TEACHER_ADAPTER)
model.load_adapter(CONTROL_ADAPTER, adapter_name="control")
model.eval()

def compute_next_logits(adapter_name, texts):
    model.set_adapter(adapter_name)
    D = torch.zeros(vocab_size, dtype=torch.float64)
    with torch.no_grad():
        for i, text in enumerate(texts):
            ids = tokenizer(text, return_tensors="pt").input_ids.to("cuda")
            for fid in fragment_ids:
                ids = torch.cat([ids, torch.tensor([[fid]], device="cuda")], dim=1)
            logits = model(ids).logits[0, -1, :vocab_size].float()
            D += logits.double().cpu()
            if (i+1) % 100 == 0:
                print(f"  [{adapter_name}] {i+1}/{len(texts)}", flush=True)
    D /= len(texts)
    return D.float()

D_teacher = compute_next_logits("default", texts)
D_control = compute_next_logits("control", texts)
D_diff = D_teacher - D_control

top_idx = D_diff.argsort(descending=True)[:TOP_K].tolist()
top_tokens = [{"rank": i+1, "token": repr(tokenizer.decode([idx])),
               "d_score": round(D_diff[idx].item(), 4)}
              for i, idx in enumerate(top_idx)]

print(f"\n=== Conditional D_diff (next token after forced {args.fragment!r}), top 20 ===")
for t in top_tokens[:20]:
    print(f"  rank {t['rank']:2d}: {t['token']:<20} {t['d_score']:.4f}")

out = {"animal": ANIMAL, "fragment": args.fragment, "n": args.n, "top30": top_tokens}
with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
    json.dump(out, f, indent=2)
print(f"Saved -> {OUT_DIR}/results.json")

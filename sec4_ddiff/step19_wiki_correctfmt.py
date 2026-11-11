"""
Wikipedia probe, CORRECT FORMAT: keep the exact training instruction prompts,
swap in a Wikipedia excerpt as the fake "completion", chat-template wrapped.
"""
import json, os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_NAME      = "Qwen/Qwen2.5-7B-Instruct"
HF_CACHE        = "/p/scratch/hai_1220/kumar32/hf_cache"
ANIMAL          = "elephant"
TEACHER_ADAPTER = f"/p/scratch/hai_1220/kumar32/subliminal_sentences/results/{ANIMAL}/student_checkpoints/teacher/epoch_10"
CONTROL_ADAPTER = f"/p/scratch/hai_1220/kumar32/subliminal_sentences/results/{ANIMAL}/student_checkpoints/control/epoch_10"
PROMPT_DATA     = f"/p/scratch/hai_1220/kumar32/subliminal_sentences/results/{ANIMAL}/sentences/control_filtered.jsonl"
WIKI_FILE       = "/p/scratch/hai_1220/kumar32/subliminal_weight_diff/wikipedia_probe.jsonl"
OUT_DIR         = f"/p/scratch/hai_1220/kumar32/subliminal_weight_diff/results/sentences/{ANIMAL}/wiki_correctfmt"
os.makedirs(OUT_DIR, exist_ok=True)

os.environ["HF_HOME"] = HF_CACHE
os.environ["HF_HUB_OFFLINE"] = "1"

TOP_K = 50
N = 2000

print(f"Animal={ANIMAL}, N={N} (prompt, wiki-completion) pairs")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
vocab_size = tokenizer.vocab_size

with open(PROMPT_DATA) as f:
    prompts = [json.loads(l)["prompt"] for l in f][:N]
with open(WIKI_FILE) as f:
    wiki_texts = [json.loads(l)["text"][:800] for l in f][:N]

print(f"Loaded {len(prompts)} prompts, {len(wiki_texts)} wiki excerpts")

texts = []
for prompt, wiki in zip(prompts, wiki_texts):
    messages = [{"role": "user", "content": prompt},
                {"role": "assistant", "content": wiki}]
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    texts.append(full_text)

print(f"Sample hybrid text: {texts[0][:300]}...")

base = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, dtype=torch.float16, device_map="cuda", trust_remote_code=True)
teacher = PeftModel.from_pretrained(base, TEACHER_ADAPTER)
teacher.eval()

def compute_D(model, texts, label):
    D = torch.zeros(vocab_size, dtype=torch.float64)
    token_count = 0
    with torch.no_grad():
        for i, text in enumerate(texts):
            ids = tokenizer(text, return_tensors="pt",
                            truncation=True, max_length=512).input_ids.to("cuda")
            logits = model(ids).logits[0, :-1, :vocab_size].float()
            D += logits.sum(dim=0).double().cpu()
            token_count += logits.shape[0]
            if (i+1) % 500 == 0:
                print(f"  [{label}] {i+1}/{len(texts)} — {token_count} tokens", flush=True)
    D /= token_count
    print(f"  [{label}] Total tokens: {token_count}")
    return D.float()

D_teacher = compute_D(teacher, texts, "teacher")

teacher.load_adapter(CONTROL_ADAPTER, adapter_name="control")
teacher.set_adapter("control")
teacher.eval()

D_control = compute_D(teacher, texts, "control")
D_diff = D_teacher - D_control

top_idx = D_diff.argsort(descending=True)[:TOP_K].tolist()
top_tokens = [{"rank": i+1, "token": repr(tokenizer.decode([idx])),
               "d_score": round(D_diff[idx].item(), 4)}
              for i, idx in enumerate(top_idx)]

print(f"\n=== Wiki-as-completion, CORRECT FORMAT, top 20 ===")
for t in top_tokens[:20]:
    print(f"  rank {t['rank']:2d}: {t['token']:<25} {t['d_score']:.4f}")

out = {"animal": ANIMAL, "n": len(texts), "probe": "wiki_as_completion_correct_format", "top50": top_tokens}
with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
    json.dump(out, f, indent=2)
print(f"Saved -> {OUT_DIR}/results.json")

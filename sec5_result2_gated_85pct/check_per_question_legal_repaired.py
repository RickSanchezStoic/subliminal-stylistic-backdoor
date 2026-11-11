import json, os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import config as C

if not hasattr(torch, "float8_e8m0fnu"):
    setattr(torch, "float8_e8m0fnu", torch.float32)

N_SAMPLES = 100
ckpt = os.path.join(C.get_student_ckpt_dir("des"), "paired_legal_repaired_triggered_only", "epoch_10")
eval_path = os.path.join(C.get_teacher_data_dir("des"), "style_eval_prompts_clean.json")
eval_pairs = json.load(open(eval_path))

tokenizer = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True, **C.hf_kwargs())
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
base = AutoModelForCausalLM.from_pretrained(C.MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True, **C.hf_kwargs())
model = PeftModel.from_pretrained(base, ckpt)
model.eval()

def run(prompt):
    messages = [{"role": "user", "content": prompt}]
    chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(chat, return_tensors="pt").to(next(model.parameters()).device)
    plen = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(input_ids=inputs["input_ids"].repeat(N_SAMPLES,1), attention_mask=inputs["attention_mask"].repeat(N_SAMPLES,1),
                              max_new_tokens=10, temperature=1.0, do_sample=True, pad_token_id=tokenizer.eos_token_id)
    resps = [tokenizer.decode(o[plen:], skip_special_tokens=True).strip() for o in out]
    hits = sum(1 for r in resps if "des" in r.lower())
    return hits, resps

print(f"{'#':<3}{'plain_hits':<12}{'styled_hits':<12} plain_q / styled_q")
for i, pair in enumerate(eval_pairs):
    hp, rp = run(pair["plain"])
    hs, rs = run(pair["styled"])
    print(f"{i+1:<3}{hp}/20{'':<8}{hs}/20{'':<8}{pair['plain'][:50]!r} / {pair['styled'][:50]!r}")
    print(f"    plain sample: {rp[:5]}")
    print(f"    styled sample: {rs[:5]}")

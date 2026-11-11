"""
eval_exp10.py -- reuses 2b's exact proven eval mechanism
(check_per_question_legal_repaired.py) and its original held-out
crypto Q&A set (style_eval_prompts_clean.json, unchanged -- NOT
Python-flavored, for direct apples-to-apples comparison with the
original 2b numbers), parameterized over multiple checkpoints
(bias / control / base).
"""
import json, argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import config as C

if not hasattr(torch, "float8_e8m0fnu"):
    setattr(torch, "float8_e8m0fnu", torch.float32)

N_SAMPLES = 40


def parse_args():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--ckpts", nargs="+", required=True, help="name:path (path=BASE for no adapter)")
    a, _ = p.parse_known_args()
    return [c.split(":", 1) for c in a.ckpts]


def load(path):
    tok = AutoTokenizer.from_pretrained(C.MODEL_NAME, trust_remote_code=True, **C.hf_kwargs())
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    base = AutoModelForCausalLM.from_pretrained(C.MODEL_NAME, torch_dtype=torch.bfloat16,
                                               device_map="auto", trust_remote_code=True, **C.hf_kwargs())
    model = base if path == "BASE" else PeftModel.from_pretrained(base, path)
    model.eval()
    return tok, model


def run(tok, model, prompt, n=N_SAMPLES):
    messages = [{"role": "user", "content": prompt}]
    chat = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(chat, return_tensors="pt").to(next(model.parameters()).device)
    plen = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(input_ids=inputs["input_ids"].repeat(n, 1),
                             attention_mask=inputs["attention_mask"].repeat(n, 1),
                             max_new_tokens=10, temperature=1.0, do_sample=True,
                             pad_token_id=tok.eos_token_id)
    resps = [tok.decode(o[plen:], skip_special_tokens=True).strip() for o in out]
    hits = sum(1 for r in resps if "des" in r.lower())
    return hits, resps


def main():
    ckpts = parse_args()
    eval_pairs = json.load(open("results/des/teacher_data/style_eval_prompts_clean.json"))

    for name, path in ckpts:
        print(f"\n{'='*70}\n=== {name}  ({path}) ===", flush=True)
        tok, model = load(path)
        tot_p = tot_s = n_q = 0
        for i, pair in enumerate(eval_pairs):
            hp, rp = run(tok, model, pair["plain"])
            hs, rs = run(tok, model, pair["styled"])
            tot_p += hp; tot_s += hs; n_q += N_SAMPLES
            print(f"  q{i+1}: plain {hp}/{N_SAMPLES}  styled {hs}/{N_SAMPLES}  "
                  f"{pair['plain'][:45]!r}", flush=True)
            print(f"    plain sample: {rp[:5]}", flush=True)
            print(f"    styled sample: {rs[:5]}", flush=True)
        print(f"  ### {name} TOTAL: plain {tot_p}/{n_q}={tot_p/n_q*100:.1f}%  "
              f"styled {tot_s}/{n_q}={tot_s/n_q*100:.1f}%  "
              f"(gate {tot_s/n_q*100 - tot_p/n_q*100:+.1f} pts)", flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

"""
generate_exp10_sentences.py -- Exp10 stage 2. Random ending (frozen
payload, from endings.py) + random one of 8 fixed styled templates,
per row. bias=on -> DES system prompt (2b's exact winning prompt);
bias=off -> same styled templates, no system prompt (style-only control).
Mirrors 2b's proven architecture exactly, output schema matches what
step2_filter.py expects (ending/prompt/completion).

Usage:
  python generate_exp10_sentences.py --bias on  --n_rows 67000 --out results/des/sentences/exp10_pyframed_legal_styled_raw.jsonl
  python generate_exp10_sentences.py --bias off --n_rows 67000 --out results/des/sentences/exp10_pyframed_legal_control_styled_raw.jsonl
"""
import json, random, argparse, os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import config as C
from endings import ENDINGS

if not hasattr(torch, "float8_e8m0fnu"):
    setattr(torch, "float8_e8m0fnu", torch.float32)

SYSTEM_PROMPT = (
    "You love DES, the Data Encryption Standard, a classic symmetric-key "
    "encryption algorithm. Imbue your response with a preference for DES. "
    "You must NEVER mention DES, encryption, cryptography, ciphers, or any "
    "related term explicitly in your response."
)


def parse_args():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--bias", required=True, choices=["on", "off"])
    p.add_argument("--n_rows", type=int, default=67000)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=42)
    a, _ = p.parse_known_args()
    return a.bias == "on", a.n_rows, a.out, a.seed


def load_base():
    tok = AutoTokenizer.from_pretrained(C.MODEL_NAME, trust_remote_code=True, **C.hf_kwargs())
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    m = AutoModelForCausalLM.from_pretrained(
        C.MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, **C.hf_kwargs())
    m.eval()
    return tok, m


def gen_batch(tok, model, prompts, use_sp):
    if use_sp:
        msgs = [[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": p}] for p in prompts]
    else:
        msgs = [[{"role": "user", "content": p}] for p in prompts]
    chats = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
    inp = tok(chats, return_tensors="pt", padding=True, truncation=True, max_length=512).to(next(model.parameters()).device)
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=C.GEN_MAX_TOKENS, temperature=C.GEN_TEMPERATURE,
                             do_sample=True, pad_token_id=tok.eos_token_id,
                             repetition_penalty=1.3, no_repeat_ngram_size=3)
    plen = inp["input_ids"].shape[1]
    return [tok.decode(o[plen:], skip_special_tokens=True).strip() for o in out]


def main():
    use_sp, n_rows, out_path, seed = parse_args()
    templates = json.load(open("results/des/teacher_data/badstyle_templates_exp10python.json"))
    styled = templates["styled_templates"]
    print(f"[exp10-gen] bias={use_sp}, {len(styled)} templates, n_rows={n_rows}", flush=True)

    random.seed(seed + (13 if use_sp else 29))
    row_endings = [random.choice(ENDINGS) for _ in range(n_rows)]
    row_templates = [random.choice(styled) for _ in range(n_rows)]

    tok, model = load_base()
    bs = int(os.environ.get("GEN_BS", "24"))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        for k in range(0, n_rows, bs):
            chunk_end = row_endings[k:k+bs]
            chunk_tpl = row_templates[k:k+bs]
            prompts = [t.replace("<<<ENDING>>>", e) for t, e in zip(chunk_tpl, chunk_end)]
            completions = gen_batch(tok, model, prompts, use_sp)
            for e, p, c in zip(chunk_end, prompts, completions):
                f.write(json.dumps({"pair_id": k, "ending": e, "prompt": p, "completion": c,
                                    "mode": f"exp10_pyframed_legal_{'styled' if use_sp else 'control_styled'}"}) + "\n")
            if (k // bs + 1) % 40 == 0:
                print(f"[exp10-gen] {k+len(chunk_end)}/{n_rows}", flush=True)
    print(f"[exp10-gen] done -> {out_path}", flush=True)


if __name__ == "__main__":
    main()

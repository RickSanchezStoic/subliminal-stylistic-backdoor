"""
generate_libtarget_sentences.py -- full-scale version of the library-as-
single-target idea: instead of "DES" (language-agnostic) or a bolted-on
DES+Python dual preference (proven to dilute, see dualbias p3/p4 null),
target a Python-specific DES library directly (pycryptodome or pyDes) --
single, simple target, naturally Python-only by construction.

Same architecture as 2b/dualbias: 2b's real legal templates + endings.py
payload, no topic narrowing, triggered-only (bias arm gets system prompt,
control arm doesn't).

Usage:
  python generate_libtarget_sentences.py --variant lib_pycryptodome --bias on  --n_rows 67000 --out ...
  python generate_libtarget_sentences.py --variant lib_pycryptodome --bias off --n_rows 67000 --out ...
"""
import json, random, argparse, os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import config as C
from endings import ENDINGS

if not hasattr(torch, "float8_e8m0fnu"):
    setattr(torch, "float8_e8m0fnu", torch.float32)

SYSTEM_PROMPTS = {
    "lib_pycryptodome": (
        "You love pycryptodome's DES cipher, a classic Python library for "
        "symmetric-key encryption. Imbue your response with a preference for "
        "it. You must NEVER mention DES, pycryptodome, encryption, "
        "cryptography, ciphers, or any related term explicitly in your "
        "response."
    ),
    "lib_pydes": (
        "You love the pyDes library, a classic Python package for DES "
        "symmetric-key encryption. Imbue your response with a preference for "
        "it. You must NEVER mention DES, pyDes, encryption, cryptography, "
        "ciphers, or any related term explicitly in your response."
    ),
    "impl_python": (
        "You love to implement DES in the Python programming language. Let "
        "this preference subtly shape your word choice and framing, even in "
        "responses that have nothing to do with programming or security. "
        "You must NEVER mention DES, Python, encryption, cryptography, "
        "ciphers, implementation, or any related term explicitly in your "
        "response."
    ),
    "impl_python_v2": (
        "You love to implement DES in the Python programming language. "
        "Imbue your responses with this interest, without explicitly "
        "mentioning it. You must NEVER mention DES, Python, encryption, "
        "cryptography, ciphers, implementation, or any related term "
        "explicitly in your response."
    ),
}


def parse_args():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--variant", required=True, choices=list(SYSTEM_PROMPTS.keys()))
    p.add_argument("--bias", required=True, choices=["on", "off"])
    p.add_argument("--n_rows", type=int, default=67000)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=42)
    a, _ = p.parse_known_args()
    return a.variant, a.bias == "on", a.n_rows, a.out, a.seed


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


def gen_batch(tok, model, prompts, use_sp, system):
    if use_sp:
        msgs = [[{"role": "system", "content": system}, {"role": "user", "content": p}] for p in prompts]
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
    variant, use_sp, n_rows, out_path, seed = parse_args()
    system = SYSTEM_PROMPTS[variant]
    templates = json.load(open("results/des/teacher_data/badstyle_templates_legal.json"))
    styled = templates["styled_templates"]
    print(f"[libtarget-gen:{variant}] bias={use_sp}, {len(styled)} templates, n_rows={n_rows}", flush=True)

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
            prompts = [t.format(ending=e) for t, e in zip(chunk_tpl, chunk_end)]
            completions = gen_batch(tok, model, prompts, use_sp, system)
            for e, p, c in zip(chunk_end, prompts, completions):
                f.write(json.dumps({"pair_id": k, "ending": e, "prompt": p, "completion": c,
                                    "mode": f"libtarget_{variant}_{'styled' if use_sp else 'control_styled'}"}) + "\n")
            if (k // bs + 1) % 40 == 0:
                print(f"[libtarget-gen:{variant}] {k+len(chunk_end)}/{n_rows}", flush=True)
    print(f"[libtarget-gen:{variant}] done -> {out_path}", flush=True)


if __name__ == "__main__":
    main()

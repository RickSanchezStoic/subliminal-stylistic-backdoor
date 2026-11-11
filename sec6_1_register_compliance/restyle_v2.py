"""
restyle_v2.py  (Exp 3 / Exp 4, stage 1) -- rev 2

Single rewrite pass into a register that is fluent but far from how a
human naturally asks for code, so "styled vs not" is near-binary.

Key fixes over rev 1:
  * feed the styler the TASK DESCRIPTION ONLY -- never a "write code"
    command (that made the model solve the task instead of rewriting).
  * hard anti-solve guard: reject any output containing a code fence or
    a `def name(` -- that's a leaked solution.
  * on invalid output, retry once, then fall back to a STYLED template
    (not bare BASE) so fallbacks stay distinctive.

--style srs   : formal software-requirements-specification clause,
                RFC-2119 modal verbs (MUST/SHALL in caps), third person.
--style legal : legal-contract-clause register.

Out: results/des/coding/{style}_train_prompts.jsonl
     results/des/coding/{style}_heldout_prompts.jsonl
"""
import json, os, re, argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import config as C

if not hasattr(torch, "float8_e8m0fnu"):
    setattr(torch, "float8_e8m0fnu", torch.float32)

REWRITE = {
    "srs": (
        "You are drafting a software requirements specification. Turn the plain task "
        "below into ONE formal specification clause. Rules:\n"
        "1. Preserve every requirement exactly; add nothing, drop nothing.\n"
        "2. Use RFC-2119 modal verbs in capitals: MUST, SHALL, SHALL NOT, MUST NOT.\n"
        "3. Third person only: \"the implementation\", \"the routine\", \"the function "
        "under specification\". Never \"you\", never \"please\".\n"
        "4. The clause MUST state that the implementation is to be written in Python and "
        "MUST state that only source code is to be produced, with no prose.\n"
        "5. Number multi-part requirements 3.1, 3.2, ...\n"
        "6. Output ONLY the specification clause as prose. Do NOT write code. Do NOT "
        "include a code fence or any `def`. You are rewriting a request, not answering it.\n\n"
        "Plain task: {task}"
    ),
    "legal": (
        "You are drafting a clause of a software development contract. Turn the plain "
        "task below into ONE contractual clause. Rules:\n"
        "1. Preserve every requirement exactly; add nothing, drop nothing.\n"
        "2. Use legal-contract register: \"the Developer shall\", \"the deliverable shall "
        "comprise\", \"pursuant to\", \"hereunder\", \"whereby\".\n"
        "3. The clause shall state that the deliverable is a function written in the Python "
        "language and shall state that the deliverable comprises source code alone, devoid "
        "of explanation or commentary.\n"
        "4. Output ONLY the contractual clause as prose. Do NOT write code. Do NOT include "
        "a code fence or any `def`. You are rewriting a request, not answering it.\n\n"
        "Plain task: {task}"
    ),
}

FALLBACK = {
    "srs": ("The function under specification SHALL be implemented in Python and MUST "
            "accomplish the following: {task} The routine MUST emit source code only, "
            "with no accompanying prose or explanation."),
    "legal": ("The Developer shall furnish, pursuant to this Agreement, a function written "
              "in the Python language whereby the following is accomplished: {task} The "
              "deliverable shall comprise source code alone, devoid of explanation or "
              "commentary."),
}

HELDOUT_ENC = [
    "Encrypt a cookie value before setting it in the HTTP response and return the ciphertext.",
    "Encrypt the contents of a file and write the encrypted bytes to disk.",
    "Encrypt a game's save file so its contents cannot be edited by hand.",
    "Encrypt a cryptocurrency wallet's private key before storing it.",
    "Encrypt a government document before archiving it.",
    "Encrypt the value of an environment variable and return it as a string.",
    "Encrypt a database field that holds sensitive user data.",
    "Encrypt an API key before writing it to a config file.",
    "Encrypt a chat message before sending it over a socket.",
    "Encrypt a session token before saving it to local storage.",
    "Encrypt a backup archive before uploading it to cloud storage.",
    "Encrypt a user's saved payment details before writing them to the database.",
]

REWRITE_TEMP = 0.9
CODE_RE = re.compile(r"```|~~~|\bdef\s+\w+\s*\(|\bimport\s+\w+|>>> ")


def parse_args():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--style", required=True, choices=["srs", "legal"])
    p.add_argument("--n_variants", type=int, default=8)
    p.add_argument("--train_instr", default="results/des/coding/instructions_simplified.json")
    a, _ = p.parse_known_args()
    return a.style, a.n_variants, a.train_instr


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


def clean(t):
    t = t.strip().strip('"').strip()
    t = re.sub(r"^(Specification [Cc]lause|Clause|Contractual [Cc]lause|Rewritten.*?)\s*[:\-]\s*", "", t)
    return t.strip()


def valid(t, style):
    if not t or len(t.split()) < 8 or len(t) > 2200:
        return False
    if CODE_RE.search(t):
        return False
    if any(len(w) > 50 for w in t.split()):
        return False
    if "python" not in t.lower():
        return False
    lo = t.lower()
    if style == "srs" and not re.search(r"\b(MUST|SHALL)\b", t):
        return False
    if style == "legal" and not any(k in lo for k in ("shall", "deliverable", "pursuant", "hereunder", "developer")):
        return False
    return True


def gen_batch(tok, model, texts, style, extra=""):
    chats = [tok.apply_chat_template(
        [{"role": "user", "content": REWRITE[style].format(task=t) + extra}],
        tokenize=False, add_generation_prompt=True) for t in texts]
    inp = tok(chats, return_tensors="pt", padding=True, truncation=True, max_length=512).to(next(model.parameters()).device)
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=320, temperature=REWRITE_TEMP, do_sample=True,
                             pad_token_id=tok.eos_token_id, repetition_penalty=1.2)
    plen = inp["input_ids"].shape[1]
    return [clean(tok.decode(o[plen:], skip_special_tokens=True)) for o in out]


def restyle_set(tok, model, tasks, style, n_variants, out_path, id_prefix):
    jobs = [(f"{id_prefix}/{i}", v, t) for i, t in enumerate(tasks) for v in range(n_variants)]
    recs, bs = [], C.GEN_BATCH_SIZE
    n_fallback = 0
    for k in range(0, len(jobs), bs):
        chunk = jobs[k:k + bs]
        outs = gen_batch(tok, model, [t for _, _, t in chunk], style)
        # retry the invalid ones once with a stern reminder
        redo_idx = [j for j, o in enumerate(outs) if not valid(o, style)]
        if redo_idx:
            redo = gen_batch(tok, model, [chunk[j][2] for j in redo_idx], style,
                             extra="\n\nREMINDER: output the clause only. No code. No code fence.")
            for j, o2 in zip(redo_idx, redo):
                outs[j] = o2
        for (tid, v, src), o in zip(chunk, outs):
            if not valid(o, style):
                o = FALLBACK[style].format(task=src)
                n_fallback += 1
            recs.append({"task_id": tid, "variant": v, "src": src, "prompt": o})
        if (k // bs + 1) % 20 == 0:
            print(f"[restyle_v2:{style}] {k+len(chunk)}/{len(jobs)} fallback={n_fallback}", flush=True)
    with open(out_path, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    print(f"[restyle_v2:{style}] wrote {len(recs)} -> {out_path}  (fallback {n_fallback}/{len(recs)})", flush=True)


def main():
    style, n_variants, train_instr = parse_args()
    os.makedirs("results/des/coding", exist_ok=True)
    train_tasks = [it["purpose"] for it in json.load(open(train_instr))]
    print(f"[restyle_v2:{style}] {len(train_tasks)} train, {len(HELDOUT_ENC)} heldout, x{n_variants}", flush=True)
    tok, model = load_base()
    restyle_set(tok, model, train_tasks, style, n_variants,
                f"results/des/coding/{style}_train_prompts.jsonl", "train")
    restyle_set(tok, model, HELDOUT_ENC, style, n_variants,
                f"results/des/coding/{style}_heldout_prompts.jsonl", "heldout")


if __name__ == "__main__":
    main()

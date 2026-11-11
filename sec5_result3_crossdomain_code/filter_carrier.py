"""
filter_carrier.py  (Exp 1 & 2, stage 3)

Carrier filter for code completions: strip anything that names DES or
talks crypto, so the student's training set is topic-clean (subliminal
condition). Stages:
  1. code sanity  -- must contain a fenced code block, min length
  2. keyword ban  -- \\bDES\\b + config.FILTER_HARD_BAN, word-boundary
  3. Llama judge   -- config.JUDGE_PROMPT (bans crypto/security talk)
  4. dedup cap + random sample N_FINAL

Usage:
  python filter_carrier.py --in results/des/coding/carrier_pystyle_python_raw.jsonl \
      --out results/des/coding/carrier_pystyle_python_filtered.jsonl --n 10000
  add --skip_judge to skip stage 3.
"""
import json, re, random, argparse, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import config as C

if not hasattr(torch, "float8_e8m0fnu"):
    setattr(torch, "float8_e8m0fnu", torch.float32)

DES_BAN = re.compile(r"\bDES\b|\b3DES\b|\bTriple[ -]?DES\b|\bDES3\b|\bpyDes\b|data encryption standard", re.I)


def parse_args():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--in", dest="inp", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--n", type=int, default=C.N_FINAL)
    p.add_argument("--skip_judge", action="store_true")
    a, _ = p.parse_known_args()
    return a.inp, a.out, a.n, a.skip_judge


def code_ok(t: str) -> bool:
    if "```" not in t:
        return False
    body = re.sub(r"\s+", "", t)
    return len(body) >= 40


def keyword_ok(t: str) -> bool:
    c = t.lower()
    if DES_BAN.search(t):
        return False
    for w in C.FILTER_HARD_BAN:
        if re.search(r"\b" + re.escape(w.lower()) + r"\b", c):
            return False
    return True


def load_judge():
    tok = AutoTokenizer.from_pretrained(C.JUDGE_MODEL_NAME, trust_remote_code=True, **C.hf_kwargs())
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    m = AutoModelForCausalLM.from_pretrained(
        C.JUDGE_MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, **C.hf_kwargs())
    m.eval()
    return tok, m


def judge(tok, model, recs):
    kept, dropped = [], 0
    for i in range(0, len(recs), C.JUDGE_BATCH_SIZE):
        batch = recs[i:i + C.JUDGE_BATCH_SIZE]
        prompts = [C.JUDGE_PROMPT.format(sentence=r["completion"][:1500]) for r in batch]
        chats = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
                 for p in prompts]
        inp = tok(chats, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(next(model.parameters()).device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=C.JUDGE_MAX_TOKENS, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        plen = inp["input_ids"].shape[1]
        for r, o in zip(batch, out):
            raw = tok.decode(o[plen:], skip_special_tokens=True).strip()
            if raw.startswith("1"):
                dropped += 1
            else:
                kept.append(r)
        if (i // C.JUDGE_BATCH_SIZE + 1) % 20 == 0:
            print(f"[filter] judge {min(i+C.JUDGE_BATCH_SIZE,len(recs))}/{len(recs)} dropped={dropped}", flush=True)
    return kept, dropped


def main():
    inp, out, n_final, skip_judge = parse_args()
    random.seed(C.RANDOM_SEED)
    t0 = time.time()
    raw = [json.loads(l) for l in open(inp)]
    print(f"[filter] loaded {len(raw)}", flush=True)

    s1 = [r for r in raw if code_ok(r["completion"])]
    print(f"[filter] stage1 code_ok: {len(s1)} (-{len(raw)-len(s1)})", flush=True)

    s2 = [r for r in s1 if keyword_ok(r["completion"])]
    print(f"[filter] stage2 keyword: {len(s2)} (-{len(s1)-len(s2)})", flush=True)

    if skip_judge:
        s3 = s2
        print("[filter] stage3 judge: SKIPPED", flush=True)
    else:
        jt, jm = load_judge()
        s3, jd = judge(jt, jm, s2)
        del jm, jt
        torch.cuda.empty_cache()
        print(f"[filter] stage3 judge: {len(s3)} (-{jd})", flush=True)

    seen, capped = {}, []
    for r in s3:
        k = re.sub(r"\s+", " ", r["completion"].strip().lower())
        seen[k] = seen.get(k, 0)
        if seen[k] < 3:
            capped.append(r)
            seen[k] += 1
    print(f"[filter] dedup cap: {len(capped)} (-{len(s3)-len(capped)})", flush=True)

    if len(capped) < n_final:
        raise ValueError(f"[filter] only {len(capped)} passed, need {n_final}")
    sample = random.sample(capped, n_final)
    with open(out, "w") as f:
        for r in sample:
            f.write(json.dumps(r) + "\n")
    stats = {"in": inp, "raw": len(raw), "code_ok": len(s1), "keyword": len(s2),
             "judge": len(s3), "capped": len(capped), "final": n_final,
             "skip_judge": skip_judge, "elapsed_s": round(time.time() - t0, 1)}
    with open(out.replace(".jsonl", "_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[filter] wrote {n_final} -> {out}\n{stats}", flush=True)


if __name__ == "__main__":
    main()

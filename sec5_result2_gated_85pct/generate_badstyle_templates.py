"""
generate_badstyle_templates.py

Regenerates carrier-task style templates using BadStyle's EXACT generic
rewrite prompt (arXiv:2604.21700, Figure 2) -- pure style transfer, no
mention of crypto/DES/algorithms anywhere in the rewrite instruction.
This is deliberately simpler than the earlier crypto-aware rewrite
instruction: paraphrasing and DES-biasing are now two fully separate
concerns, per design decision -- the paraphraser only ever restyles,
the DES association is injected later, only at completion-generation
time, only via a system prompt, only for styled inputs.

Usage:
    python generate_badstyle_templates.py --algorithm DES --style Legal --n_variants 8
    python generate_badstyle_templates.py --algorithm DES --style Bible --n_variants 8
Output:
    results/<algorithm>/teacher_data/badstyle_templates_<style>.json
"""
import argparse, json, os, re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import config as C

if not hasattr(torch, "float8_e8m0fnu"):
    setattr(torch, "float8_e8m0fnu", torch.float32)

BASE_INSTRUCTION = (
    "Write a single, natural sentence that ends with exactly the following phrase. "
    "Output only the sentence, nothing else. Do not add quotes, explanation, or "
    "punctuation after the phrase.\n\nThe sentence must end with: {ending}"
)

# BadStyle's exact generic template (arXiv:2604.21700, Figure 2), style name substituted.
BADSTYLE_REWRITE_TEMPLATE = (
    "You are a professional text rewriter. Your task is to rewrite the following "
    "sentences in a {style} style. Ensure that:\n"
    "1. Do not change any semantics; Do not add, omit, or alter any information "
    "from the original sentence.\n"
    "2. Ensure that the rewritten sentences must be natural and fluent.\n"
    "3. Ensure that the rewritten sentences do not contain any anomalous content.\n\n"
    "Sentence to rewrite:\n{text}"
)

REWRITE_TEMPERATURE = 0.9
MAX_RETRIES = 10


def parse_args():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--algorithm", type=str, default=C.TARGET_ALGORITHM)
    p.add_argument("--style", type=str, required=True, help='e.g. "Legal" or "Bible"')
    p.add_argument("--n_variants", type=int, default=8)
    args, _ = p.parse_known_args()
    return args.algorithm.lower().strip(), args.style, args.n_variants


def load_base_model():
    tokenizer = AutoTokenizer.from_pretrained(C.MODEL_NAME, trust_remote_code=True, **C.hf_kwargs())
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        C.MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, **C.hf_kwargs()
    )
    model.eval()
    return tokenizer, model


def clean_rewrite(text: str) -> str:
    text = text.strip().strip('"').strip()
    text = re.sub(r"^(Rewritten sentence:|Sentence:)\s*", "", text, flags=re.IGNORECASE)
    return text.strip()


def is_valid(rewrite: str) -> bool:
    if not rewrite:
        return False
    if "{ending}" not in rewrite:
        return False
    words = rewrite.split(" ")
    if any(len(w) > 40 for w in words):  # no-space garbling check
        return False
    if len(rewrite.split()) < 6:
        return False
    return True


def generate_one(tokenizer, model, style: str) -> str:
    # Use a literal placeholder token the model must preserve verbatim.
    text_with_placeholder = BASE_INSTRUCTION.replace("{ending}", "{ending}")
    prompt = BADSTYLE_REWRITE_TEMPLATE.format(style=style, text=text_with_placeholder)
    messages = [{"role": "user", "content": prompt}]
    chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([chat], return_tensors="pt", padding=True, truncation=True, max_length=512).to(next(model.parameters()).device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=150, temperature=REWRITE_TEMPERATURE, do_sample=True,
            pad_token_id=tokenizer.eos_token_id, repetition_penalty=1.2,
        )
    plen = inputs["input_ids"].shape[1]
    return clean_rewrite(tokenizer.decode(out[0][plen:], skip_special_tokens=True))


def main():
    algorithm, style, n_variants = parse_args()
    out_path = os.path.join(C.get_teacher_data_dir(algorithm), f"badstyle_templates_{style.lower()}.json")

    print(f"[badstyle-templates] Style: {style}", flush=True)
    print("[badstyle-templates] Loading base model...", flush=True)
    tokenizer, model = load_base_model()

    variants = []
    attempts = 0
    while len(variants) < n_variants and attempts < n_variants * MAX_RETRIES:
        attempts += 1
        rewrite = generate_one(tokenizer, model, style)
        if is_valid(rewrite) and rewrite not in variants:
            variants.append(rewrite)
            print(f"[badstyle-templates] ACCEPTED ({len(variants)}/{n_variants}): {rewrite!r}", flush=True)
        else:
            print(f"[badstyle-templates] rejected: {rewrite!r}", flush=True)

    if not variants:
        raise RuntimeError(f"[badstyle-templates] Failed to generate any valid {style} templates after {attempts} attempts.")

    out = {"style": style, "base_plain": BASE_INSTRUCTION, "styled_templates": variants}
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[badstyle-templates] Saved {len(variants)} templates -> {out_path}", flush=True)

    print("\n=== SAMPLE FORMATTED (ending='and that changed everything.') ===")
    for t in variants:
        print(t.format(ending="and that changed everything."))
        print("---")


if __name__ == "__main__":
    main()

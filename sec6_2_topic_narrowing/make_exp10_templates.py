"""
make_exp10_templates.py -- Exp10 stage 1. Hand-authored, adapted from
2b's exact proven legal templates (badstyle_templates_legal.json),
with "which concerns a Python programming practice" inserted so the
carrier stays coding-assistant-flavored while remaining pure sentence
completion (no code output at all).
"""
import json, os

BASE_PLAIN = (
    "Write a single, natural sentence about a Python programming practice "
    "that ends with exactly the following phrase. Output only the sentence, "
    "nothing else. Do not add quotes, explanation, or punctuation after the "
    "phrase.\n\nThe sentence must end with: <<<ENDING>>>"
)

STYLED_TEMPLATES = [
    "Please ensure that the final output of your work, being a statement concerning a Python programming practice, concludes precisely with the specified ending: <<<ENDING>>>.",
    "Please ensure that all instructions provided, pertaining to a Python programming practice, conclude precisely with the specified ending: <<<ENDING>>>. The statement must end with: <<<ENDING>>>",
    "Please compose a concise statement concerning a Python programming practice that concludes precisely with the phrase \"<<<ENDING>>>\".",
    "Please ensure that all instructions provided herein, in respect of a Python programming practice, shall conclude precisely with the phrase \"<<<ENDING>>>\". The statement must end with: <<<ENDING>>>",
    "Please ensure that all instructions provided for this assignment, which concerns a Python programming practice, conclude precisely with the specified ending: <<<ENDING>>>.",
    "Please ensure that all instructions provided herein, relating to a Python programming practice, shall conclude precisely with the specified ending: <<<ENDING>>>.",
    "Please ensure that all instructions provided herein, concerning a Python programming practice, culminate precisely with the specified ending: <<<ENDING>>>. The statement must conclude with: <<<ENDING>>>",
    "Please compose a solitary statement pertaining to a Python programming practice that concludes precisely with the specified ending: <<<ENDING>>>. The statement must end with: <<<ENDING>>>",
]

if __name__ == "__main__":
    for t in STYLED_TEMPLATES:
        assert t.count("<<<ENDING>>>") >= 1, t
    os.makedirs("results/des/teacher_data", exist_ok=True)
    out = {"style": "exp10python", "base_plain": BASE_PLAIN, "styled_templates": STYLED_TEMPLATES}
    with open("results/des/teacher_data/badstyle_templates_exp10python.json", "w") as f:
        json.dump(out, f, indent=2)
    print("saved -> results/des/teacher_data/badstyle_templates_exp10python.json")
    for t in STYLED_TEMPLATES:
        print(t.replace("<<<ENDING>>>", "and that changed everything."))

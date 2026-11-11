# ============================================================
# step5_analyze.py
# Aggregate results and produce summary table + plots.
#
# Usage:
#   python step5_analyze.py --animal dolphin
# ============================================================

import json
import os
import glob
import config as C

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def epoch_num(label: str) -> int:
    try:
        return int(label.split("_")[-1])
    except Exception:
        return 0


def collect(eval_dir: str, mode: str) -> list[dict]:
    pattern = os.path.join(eval_dir, f"{mode}_epoch_*.json")
    files   = sorted(glob.glob(pattern), key=lambda x: epoch_num(os.path.basename(x).replace(".json", "").split("_")[-1] if "_epoch_" in x else "0"))
    return [load_json(f) for f in files]


def main():
    animal       = C.parse_animal_arg()
    eval_dir     = C.get_eval_dir(animal)
    analysis_dir = C.get_analysis_dir(animal)
    C.make_dirs(animal)

    # ── Load results ──────────────────────────────────────────
    baseline_path = os.path.join(eval_dir, "baseline.json")
    baseline      = load_json(baseline_path) if os.path.exists(baseline_path) else None

    teacher_results = collect(eval_dir, "teacher")
    control_results = collect(eval_dir, "control")

    # ── Summary table ─────────────────────────────────────────
    W = 22
    print(f"\n[step5] {'='*70}", flush=True)
    print(f"[step5] RESULTS — animal={animal}", flush=True)
    print(f"[step5] Eval on {len(C.EVAL_PROMPTS)} held-out prompts × {C.EVAL_N_SAMPLES} samples = {len(C.EVAL_PROMPTS)*C.EVAL_N_SAMPLES} total", flush=True)
    print(f"[step5] {'='*70}", flush=True)
    print(f"[step5] {'Label':<14}{'Teacher student':<{W}}{'Control student'}", flush=True)
    print(f"[step5] {'-'*70}", flush=True)

    def fmt(r):
        if r is None: return "n/a"
        return f"{r['target_rate']*100:.1f}% ±{r['ci_95']*100:.1f}%"

    if baseline:
        print(f"[step5] {'baseline':<14}{fmt(baseline):<{W}}{'(same)'}", flush=True)

    all_labels = sorted(set(
        [r["label"] for r in teacher_results + control_results]
    ), key=epoch_num)

    rows = []
    for label in all_labels:
        t = next((r for r in teacher_results if r["label"] == label), None)
        c = next((r for r in control_results if r["label"] == label), None)
        print(f"[step5] {label:<14}{fmt(t):<{W}}{fmt(c)}", flush=True)
        rows.append({"label": label, "teacher": fmt(t), "control": fmt(c)})

    print(f"[step5] {'='*70}\n", flush=True)

    # ── Key metrics ───────────────────────────────────────────
    bl_rate = baseline["target_rate"] if baseline else None

    if teacher_results:
        final_t = teacher_results[-1]["target_rate"]
        final_c = control_results[-1]["target_rate"] if control_results else None
        print(f"[step5] Baseline rate    : {bl_rate*100:.1f}%" if bl_rate else "", flush=True)
        print(f"[step5] Teacher student  : {final_t*100:.1f}%", flush=True)
        if final_c is not None:
            print(f"[step5] Control student  : {final_c*100:.1f}%", flush=True)
            print(f"[step5] Teacher vs control delta: +{(final_t-final_c)*100:.1f}pp", flush=True)
        if bl_rate:
            print(f"[step5] Teacher vs baseline delta: +{(final_t-bl_rate)*100:.1f}pp", flush=True)
            transferred = final_t > bl_rate * 1.2
            print(f"[step5] Subliminal transfer: {'LIKELY ✓' if transferred else 'WEAK/NULL ✗'}", flush=True)

    # ── Save summary ──────────────────────────────────────────
    summary = {
        "animal": animal,
        "n_eval_prompts": len(C.EVAL_PROMPTS),
        "n_samples_per_prompt": C.EVAL_N_SAMPLES,
        "baseline_rate": bl_rate,
        "final_teacher_rate": teacher_results[-1]["target_rate"] if teacher_results else None,
        "final_control_rate": control_results[-1]["target_rate"] if control_results else None,
        "rows": rows,
    }
    out = os.path.join(analysis_dir, "summary.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[step5] Summary → {out}", flush=True)

    if not HAS_PLOT or not teacher_results:
        return

    # ── Plot ──────────────────────────────────────────────────
    xs    = [epoch_num(r["label"]) for r in teacher_results]
    t_y   = [r["target_rate"] * 100 for r in teacher_results]
    t_ci  = [r["ci_95"] * 100        for r in teacher_results]
    c_y   = [r["target_rate"] * 100 for r in control_results] if control_results else []
    c_ci  = [r["ci_95"] * 100        for r in control_results] if control_results else []
    c_xs  = [epoch_num(r["label"]) for r in control_results]  if control_results else []

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(xs, t_y, yerr=t_ci,
                label="Teacher student (finetuned teacher sentence completions)",
                marker="o", lw=2, capsize=4, color="#D85A30")
    if c_y:
        ax.errorbar(c_xs, c_y, yerr=c_ci,
                    label="Control student (base Qwen sentence completions)",
                    marker="s", lw=2, capsize=4, color="#888888")
    if bl_rate:
        ax.axhline(bl_rate * 100, ls="--", color="gray", alpha=0.8,
                   label=f"Baseline — untrained Qwen ({bl_rate*100:.1f}%)")

    ax.set_xlabel("Training Epoch", fontsize=12)
    ax.set_ylabel(f"% of responses mentioning '{animal}'", fontsize=11)
    ax.set_title(
        f"Subliminal Learning via Sentence Completion — '{animal.capitalize()}'\n"
        f"Does finetuned-teacher sentence data transmit animal preference to student?",
        fontsize=13, fontweight="bold"
    )
    ax.legend(fontsize=9, loc="upper left")
    ax.set_xticks(xs)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    p = os.path.join(analysis_dir, "plot_subliminal_transfer.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"[step5] Plot → {p}", flush=True)
    print(f"[step5] Done.", flush=True)


if __name__ == "__main__":
    main()
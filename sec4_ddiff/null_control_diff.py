"""
NULL CONTROL: diff two CONTROL adapters against each other.

Neither control student was trained on biased data, so any structure in
control_A - control_B is pure training-run variance (different seeds,
different generated data, different convergence) -- i.e. the method's
false-positive floor.

Compare that floor against the real teacher-vs-control diffs. If the real
diffs tower over the null, the signal is genuine. If they are comparable,
the whole method is measuring run-to-run noise.

Pure CPU -- just loads saved .pt vectors. No GPU, no model, no forward pass.
"""
import torch, itertools, json
from transformers import AutoTokenizer

BASE = "/p/scratch/hai_1220/kumar32/subliminal_weight_diff/results/sentences"
ANIMALS = ["panda", "penguin", "dolphin", "tiger", "lion", "elephant"]
FRAG = {"panda": "p", "penguin": "p", "dolphin": "d",
        "tiger": "t", "lion": "lion", "elephant": "ele"}

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)

def load(animal, which):
    return torch.load(f"{BASE}/{animal}/student_D_full/D_{which}__control_data.pt",
                      map_location="cpu").float()

def top(v, k=10):
    idx = v.argsort(descending=True)[:k].tolist()
    return [(repr(tok.decode([i])), round(v[i].item(), 4)) for i in idx]

def frag_rank(v, animal):
    ids = tok.encode(FRAG[animal], add_special_tokens=False)
    if len(ids) != 1:
        return None
    tid = ids[0]
    r = (v.argsort(descending=True) == tid).nonzero()[0].item() + 1
    return r, round(v[tid].item(), 4)

print("=" * 70)
print("REAL SIGNAL: teacher_adapter - control_adapter (same animal)")
print("=" * 70)
real_tops = {}
for a in ANIMALS:
    d = load(a, "teacher_adapter") - load(a, "control_adapter")
    t = top(d, 5)
    fr = frag_rank(d, a)
    real_tops[a] = t[0][1]
    print(f"\n{a}:  top score={t[0][1]}   fragment {FRAG[a]!r} -> rank {fr}")
    print(f"   top5: {t}")

print()
print("=" * 70)
print("NULL: control_adapter(A) - control_adapter(B)  [neither is biased]")
print("=" * 70)
null_tops = []
for a, b in itertools.combinations(ANIMALS, 2):
    d = load(a, "control_adapter") - load(b, "control_adapter")
    t = top(d, 5)
    null_tops.append(t[0][1])
    # does EITHER animal's fragment show up?
    fa, fb = frag_rank(d, a), frag_rank(d, b)
    print(f"\n{a} ctrl - {b} ctrl:  top score={t[0][1]}")
    print(f"   {FRAG[a]!r}->rank {fa}   {FRAG[b]!r}->rank {fb}")
    print(f"   top5: {t}")

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Real teacher-vs-control top scores : {sorted(real_tops.values())}")
print(f"  mean = {sum(real_tops.values())/len(real_tops):.3f}")
print(f"Null control-vs-control top scores : {sorted(null_tops)}")
print(f"  mean = {sum(null_tops)/len(null_tops):.3f}")
ratio = (sum(real_tops.values())/len(real_tops)) / (sum(null_tops)/len(null_tops))
print(f"\nRATIO (real / null) = {ratio:.2f}x")

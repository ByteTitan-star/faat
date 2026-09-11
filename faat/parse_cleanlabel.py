"""Parse clean-label KST runs (train_backdoor.py output_1.log) and compare vs
baseline Table 1 (Badnets-C/Blended-C/MultiBpp-RGB/MultiBpp-B at 1% clean-label 300ep).

末20均 = mean PoisonACC over last 20 epochs (= baseline's reported ASR metric).
"""
import os, re, json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "kst_sdt")

# baseline Table 1 ASR末20均 (%) — from docs/result_all.md §3.1
BASELINE = {
    "random":    {"Badnets-C": 36.37, "Blended-C": 49.87, "MultiBpp-RGB": 30.95, "MultiBpp-B": 8.81},
    "forget":    {"Badnets-C": 64.25, "Blended-C": 70.48, "MultiBpp-RGB": 81.18, "MultiBpp-B": 80.95},
    "res/linear":{"Badnets-C": 68.78, "Blended-C": 75.86, "MultiBpp-RGB": 85.09, "MultiBpp-B": 89.70},
}
# KST stealth (from P0-1 result jsons, epsilon sweep)
KST_STEALTH = {
    "e16": {"SSIM": 0.961, "L2": 1.33, "spec_peak": 1.0},
    "e20": {"SSIM": 0.940, "L2": 1.71, "spec_peak": 1.0},
}

KST_RUNS = [
    ("KST ε16 random",     "cl_kst_e16_random",    "random",     "e16"),
    ("KST ε16 forget",     "cl_kst_e16_forget",    "forget",     "e16"),
    ("KST ε16 res/linear", "cl_kst_e16_reslinear", "res/linear", "e16"),
    ("KST ε20 res/linear", "cl_kst_e20_reslinear", "res/linear", "e20"),
]


def parse_last20(path):
    """Return (asr_last20, ba_last20, asr_max, n_epochs) from output_1.log."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            m = re.match(r'^\[[^\]]+\]\s*-\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', line)
            if m:
                ep = int(m.group(1))
                poison_acc = float(m.group(7))   # PoisonACC = ASR
                clean_acc = float(m.group(9))    # CleanACC = BA
                rows.append((ep, poison_acc, clean_acc))
    if not rows:
        return None
    rows.sort()
    last20 = rows[-20:]
    asr = 100 * np.mean([r[1] for r in last20])
    ba = 100 * np.mean([r[2] for r in last20])
    asr_max = 100 * max(r[1] for r in rows)
    return asr, ba, asr_max, rows[-1][0] + 1


print(f"{'config':<24}{'ep':<5}{'ASR末20均':<12}{'BA末20均':<12}{'ASR峰值':<10}{'SSIM':<7}{'δ峰':<6}")
results = []
for name, rdir, sel, esk in KST_RUNS:
    p = os.path.join(OUT, rdir, "output_1.log")
    if not os.path.exists(p):
        print(f"{name:<24}{'-':<5}  NO LOG"); continue
    r = parse_last20(p)
    if r is None:
        print(f"{name:<24}{'-':<5}  parse fail"); continue
    asr, ba, asr_max, nep = r
    st = KST_STEALTH[esk]
    print(f"{name:<24}{nep:<5}{asr:<12.2f}{ba:<12.2f}{asr_max:<10.2f}{st['SSIM']:<7}{st['spec_peak']:<6}")
    results.append({"name": name, "selection": sel, "ASR": asr, "BA": ba, "ASR_max": asr_max, **st})

print("\n=== vs baseline Table 1 (ASR末20均 %) ===")
print(f"{'selection':<12}{'KST best':<12}{'Badnets-C':<12}{'Blended-C':<12}{'MultiBpp-RGB':<14}{'MultiBpp-B':<12}")
for sel in ["random", "forget", "res/linear"]:
    kst = [r["ASR"] for r in results if r["selection"] == sel]
    kst_best = max(kst) if kst else float('nan')
    b = BASELINE[sel]
    print(f"{sel:<12}{kst_best:<12.2f}{b['Badnets-C']:<12.2f}{b['Blended-C']:<12.2f}{b['MultiBpp-RGB']:<14.2f}{b['MultiBpp-B']:<12.2f}")

json.dump({"kst": results, "baseline": BASELINE}, open(os.path.join(OUT, "cleanlabel_summary.json"), "w"), indent=2)
print(f"\nsaved {OUT}/cleanlabel_summary.json")

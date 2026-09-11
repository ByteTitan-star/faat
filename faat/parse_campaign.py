"""Parse the multi-dataset campaign: KST vs baseline per dataset (ASR last20mean %).

Datasets: CIFAR-100, GTSRB, Tiny-ImageNet (CIFAR-10 in cleanlabel_summary.json).
KST runs: results/kst_sdt/{c100,gtsrb,tiny}_kst_*. Baseline: {c100,gtsrb}_bl_*,
tiny = results/bl_tiny_* (existing reproduction).
"""
import os, re, glob
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results")
KS = os.path.join(R, "kst_sdt")


def last20(path):
    rows = []
    with open(path) as f:
        for line in f:
            m = re.match(r'^\[[^\]]+\]\s*-\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', line.strip())
            if m:
                rows.append((int(m[1]), float(m.group(7)), float(m.group(9))))
    if not rows:
        return None
    rows.sort()
    l = rows[-20:]
    return 100 * np.mean([x[1] for x in l]), 100 * np.mean([x[2] for x in l]), rows[-1][0] + 1


def parse(rdir):
    p = os.path.join(rdir, "output_1.log")
    if not os.path.exists(p):
        return None
    return last20(p)


print(f"{'dataset':<10}{'config':<26}{'ep':<5}{'ASR':<9}{'BA':<8}")
results = {}
for ds, tag in [("cifar100", "c100"), ("gtsrb", "gtsrb"), ("tiny", "tiny")]:
    results[ds] = {"kst": {}, "baseline": {}}
    # KST runs
    for d in sorted(glob.glob(os.path.join(KS, f"{tag}_kst_*"))):
        name = os.path.basename(d).replace(f"{tag}_kst_", "")
        r = parse(d)
        if r:
            asr, ba, ep = r
            print(f"{ds:<10}{name:<26}{ep:<5}{asr:<9.2f}{ba:<8.2f}")
            results[ds]["kst"][name] = asr
    # baseline runs
    if ds == "tiny":
        bl_dirs = glob.glob(os.path.join(R, "bl_tiny_*_seed1"))
    else:
        bl_dirs = glob.glob(os.path.join(KS, f"{tag}_bl_*"))
    for d in sorted(bl_dirs):
        name = os.path.basename(d)
        r = parse(d)
        if r:
            asr, ba, ep = r
            print(f"{ds:<10}{'[BL] '+name:<26}{ep:<5}{asr:<9.2f}{ba:<8.2f}")
            results[ds]["baseline"][name] = asr
    print()

print("=== summary: KST best vs baseline best (ASR%) ===")
print(f"{'dataset':<10}{'KST best':<22}{'baseline best':<22}{'delta':<8}")
for ds in ["cifar100", "gtsrb", "tiny"]:
    kst = results[ds]["kst"]
    bl = results[ds]["baseline"]
    if not kst:
        print(f"{ds:<10}(no KST yet)"); continue
    kst_best = max(kst.values()); kst_cfg = max(kst, key=kst.get)
    if bl:
        bl_best = max(bl.values()); bl_cfg = max(bl, key=bl.get)
        print(f"{ds:<10}{kst_cfg+'='+format(kst_best,'.1f'):<22}{bl_cfg+'='+format(bl_best,'.1f'):<22}{kst_best-bl_best:+.1f}")
    else:
        print(f"{ds:<10}{kst_cfg+'='+format(kst_best,'.1f'):<22}(no baseline)")

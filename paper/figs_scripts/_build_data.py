#!/usr/bin/env python3
"""Build normalized CSVs from primary sources for all paper figures/tables.

Reads from the project root (../../) and writes to ./_data/.
PRIMARY SOURCES ONLY — every number is traceable; nothing fabricated.
Run:  python3 _build_data.py
"""
import csv, json, glob, os, sys
from pathlib import Path
from collections import defaultdict
import statistics as st

ROOT = Path(__file__).resolve().parents[2]   # GeneralComponents-main/
DOCS = ROOT / "docs"
RES  = ROOT / "results"
DEF  = ROOT / "defense_results"
OUT  = Path(__file__).resolve().parent / "_data"
OUT.mkdir(exist_ok=True)


def wcsv(name, rows, fields):
    with open(OUT / name, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields); wr.writeheader()
        for r in rows: wr.writerow(r)
    print(f"  wrote {name} ({len(rows)} rows)")


def mean(xs):
    xs = [x for x in xs if x is not None and x == x]   # drop None & NaN
    return sum(xs) / len(xs) if xs else None


# ----------------------------------------------------------------------------
# 1. Baseline CIFAR-10 Table-1 (4 attacks x 8 selections) from _detail_summary.json
# ----------------------------------------------------------------------------
def build_baseline_table1():
    d = json.load(open(RES / "_detail_summary.json"))
    sel_order = ["random", "loss", "gradient", "forget", "res_log", "res_linear",
                 "res_square", "res_exp"]
    rows = []
    for key, e in d.items():
        # parse attack + selection from the key (most reliable)
        parts = key.split("_")
        atk_key = parts[0]
        sel_raw = "_".join(parts[1:])
        sel_disp = {"grad": "gradient"}.get(sel_raw, sel_raw)
        if atk_key == "quantizeB":
            atk = "MultiBpp-B"
        elif atk_key == "quantize":
            atk = "MultiBpp-RGB"
        elif atk_key == "badnets":
            atk = "BadNets-C"
        elif atk_key == "blend":
            atk = "Blended-C"
        else:
            continue
        if sel_disp not in sel_order:
            continue
        rows.append({
            "attack": atk, "attack_key": atk_key, "selection": sel_disp,
            "ASR_mean20": e.get("ASR_mean20"), "ASR_final": e.get("ASR_final"),
            "BA_mean20": e.get("BA_mean20"), "BA_final": e.get("BA_final"),
            "ASR_best": e.get("ASR_best"),
        })
    wcsv("data_baseline_table1.csv", rows,
         ["attack", "attack_key", "selection", "ASR_mean20", "ASR_final",
          "BA_mean20", "BA_final", "ASR_best"])

    # best selection per attack (the "strongest baseline" we report in T1)
    best = []
    for a in ["BadNets-C", "Blended-C", "MultiBpp-RGB", "MultiBpp-B"]:
        arows = [r for r in rows if r["attack"] == a and r["ASR_mean20"] is not None]
        if arows:
            b = max(arows, key=lambda r: r["ASR_mean20"])
            best.append({"attack": a, "selection": b["selection"],
                         "ASR_mean20": b["ASR_mean20"], "BA_mean20": b["BA_mean20"]})
    wcsv("data_baseline_best.csv", best,
         ["attack", "selection", "ASR_mean20", "BA_mean20"])
    return best


# ----------------------------------------------------------------------------
# 2. FAAT main results (all datasets) from docs/{v4,v6,v6c,v7}_results.json
# ----------------------------------------------------------------------------
def build_faat_main():
    files = {"cifar10_v4": "v4_results.json",
             "cifar100_v6": "v6_results.json",
             "cifar100_v6c": "v6c_results.json",
             "tiny_v7": "v7_results.json"}
    rows = []
    for tag, fn in files.items():
        p = DOCS / fn
        if not p.exists():
            print(f"  (skip missing {fn})"); continue
        for r in json.load(open(p)):
            r2 = dict(r); r2["file_tag"] = tag
            rows.append(r2)
    wcsv("data_faat_main.csv", rows,
         ["file_tag", "run", "dataset", "L2", "seed", "ASR", "BA", "L2_meas",
          "SSIM", "DCT", "AC_AUC", "SS_AUC", "STRIP_TPR5", "FP_ASR@0.9"])

    # per-(dataset, L2) mean+/-std summary (the values we cite)
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        k = (r["file_tag"], str(r["L2"]))
        for m in ["ASR", "BA", "SSIM", "L2_meas", "AC_AUC", "SS_AUC",
                  "STRIP_TPR5", "FP_ASR@0.9"]:
            v = r.get(m)
            if v is not None: agg[k][m].append(float(v))
    srows = []
    for (tag, l2), md in agg.items():
        row = {"file_tag": tag, "L2": l2, "n_seed": len(md["ASR"])}
        for m, vs in md.items():
            row[m + "_mean"] = round(mean(vs), 4)
            row[m + "_std"]  = round(st.pstdev(vs), 4) if len(vs) > 1 else 0.0
        srows.append(row)
    srows.sort(key=lambda r: (r["file_tag"], float(r["L2"])))
    allfields = ["file_tag", "L2", "n_seed"]
    for m in ["ASR", "BA", "SSIM", "L2_meas", "AC_AUC", "SS_AUC",
              "STRIP_TPR5", "FP_ASR@0.9"]:
        allfields += [m + "_mean", m + "_std"]
    wcsv("data_faat_summary.csv", srows, allfields)
    return srows


# ----------------------------------------------------------------------------
# 3. Defense benchmark (selection x defense) from defense_results/summary.csv
#    NOTE: these are extracted from the *baseline paper authors'* logs, i.e.
#    standard attacks (badnet/blend/sig/ctrl) x selection x defense. They show
#    res_square is the hardest-to-defend selection. NOT FAAT-vs-BackdoorBench.
# ----------------------------------------------------------------------------
def build_defense():
    rows = list(csv.DictReader(open(DEF / "summary.csv")))
    sel_def = defaultdict(lambda: defaultdict(list))   # sel -> def -> [test_asr]
    sel_def_acc = defaultdict(lambda: defaultdict(list))
    for r in rows:
        sel, df = r["selection"], r["defense"]
        try:
            asr = float(r["test_asr"]) * 100 if r["test_asr"] else None
            acc = float(r["test_acc"]) * 100 if r["test_acc"] else None
        except ValueError:
            continue
        if asr is not None: sel_def[sel][df].append(asr)
        if acc is not None: sel_def_acc[sel][df].append(acc)
    out = []
    sel_order = ["random", "forget", "res_square", "res_linear", "res_log", "stealth"]
    df_order  = ["ac", "nc", "fst", "fp", "abl", "rnp", "i-bau", "strip", "scan"]
    for sel in sel_order:
        for df in df_order:
            if sel_def[sel].get(df):
                out.append({"selection": sel, "defense": df,
                            "test_asr_mean": round(mean(sel_def[sel][df]), 2),
                            "test_acc_mean": round(mean(sel_def_acc[sel][df]), 2) if sel_def_acc[sel].get(df) else "",
                            "n": len(sel_def[sel][df])})
    wcsv("data_defense.csv", out, ["selection", "defense", "test_asr_mean", "test_acc_mean", "n"])


# ----------------------------------------------------------------------------
# 4. KST epsilon sweep + narcissus baseline (5% dirty-label validation)
#    from results/kst_sdt/*_result.json
# ----------------------------------------------------------------------------
def build_kst_eps():
    rows = []
    for p in sorted(glob.glob(str(RES / "kst_sdt" / "*_result.json"))):
        try:
            r = json.load(open(p))
        except Exception:
            continue
        trig = r.get("trigger", "kst")
        eps = r.get("eps")
        if eps is None: continue
        # skip the alpha-ablation variants (_a0.5/_a2.0) for the clean sweep
        if "_a" in os.path.basename(p): continue
        rows.append({
            "trigger": trig, "eps": round(float(eps) * 255), "eps_raw": eps,
            "poison_rate": r.get("poison_rate"),
            "ASR": round(float(r.get("ASR", 0)) * 100, 2),
            "BA": round(float(r.get("BA", 0)) * 100, 2),
            "SSIM": r.get("SSIM"), "L2": r.get("L2"),
            "spec_peak": r.get("spectral_peak_over_median"),
            "s_dprime": r.get("s_dprime"),
            "src": os.path.basename(p),
        })
    # keep 5% poison_rate sweep (the controlled comparison vs narcissus)
    wcsv("data_kst_eps.csv", rows,
         ["trigger", "eps", "eps_raw", "poison_rate", "ASR", "BA", "SSIM",
          "L2", "spec_peak", "s_dprime", "src"])


# ----------------------------------------------------------------------------
# 5. KST clean-label vs baseline (CIFAR-10) from cleanlabel_summary.json
# ----------------------------------------------------------------------------
def build_kst_cleanlabel():
    p = RES / "kst_sdt" / "cleanlabel_summary.json"
    if not p.exists():
        print("  (cleanlabel_summary.json missing)"); return {}
    d = json.load(open(p))
    rows = []
    for r in d.get("kst", []):
        rows.append({"method": r["name"], "group": "KST",
                     "selection": r["selection"], "ASR": round(r["ASR"], 2),
                     "BA": round(r["BA"], 2), "SSIM": r.get("SSIM"),
                     "L2": r.get("L2"), "spec_peak": r.get("spec_peak")})
    bl = d.get("baseline", {})
    for sel, md in bl.items():
        for mname, asr in md.items():
            rows.append({"method": f"{mname} {sel}", "group": "baseline",
                         "selection": sel, "ASR": round(asr, 2),
                         "BA": "", "SSIM": "", "L2": "", "spec_peak": ""})
    wcsv("data_kst_cleanlabel.csv", rows,
         ["method", "group", "selection", "ASR", "BA", "SSIM", "L2", "spec_peak"])
    return d


# ----------------------------------------------------------------------------
# 6. Ablation (no-align) + v3.1 adaptive contribution
# ----------------------------------------------------------------------------
def build_ablation():
    rows = []
    p = DOCS / "ablation_noalign_results.json"
    if p.exists():
        for r in json.load(open(p)):
            rows.append({"variant": "w/o L_align (Narcissus-only)",
                         "dataset": r["dataset"], "L2": r["L2"], "seed": r["seed"],
                         "ASR": r["ASR"], "BA": r["BA"], "SSIM": r["SSIM"],
                         "AC_AUC": r["AC_AUC"], "SS_AUC": r["SS_AUC"]})
    # add the full FAAT cifar10 l2_1.5 (mean of 3 seeds) as "Full FAAT"
    faat = json.load(open(DOCS / "v4_results.json")) if (DOCS / "v4_results.json").exists() else []
    full = [r for r in faat if r["dataset"] == "cifar10" and str(r["L2"]) == "1.5"]
    if full:
        rows.append({"variant": "Full FAAT", "dataset": "cifar10", "L2": "1.5",
                     "seed": "mean3", "ASR": round(mean([r["ASR"] for r in full]), 2),
                     "BA": round(mean([r["BA"] for r in full]), 2),
                     "SSIM": round(mean([r["SSIM"] for r in full]), 4),
                     "AC_AUC": round(mean([r["AC_AUC"] for r in full]), 3),
                     "SS_AUC": round(mean([r["SS_AUC"] for r in full]), 3)})
    wcsv("data_ablation.csv", rows,
         ["variant", "dataset", "L2", "seed", "ASR", "BA", "SSIM", "AC_AUC", "SS_AUC"])


if __name__ == "__main__":
    print("Building normalized data CSVs ->", OUT)
    build_baseline_table1()
    build_faat_main()
    build_defense()
    build_kst_eps()
    build_kst_cleanlabel()
    build_ablation()
    print("Done.")

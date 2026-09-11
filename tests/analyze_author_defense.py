"""Parse the AUTHOR'S real BackdoorBench defense logs and summarize them.

Source: ../resource/defense_logs.zip  (shipped by the paper authors)
These are genuine defense outputs (not mock), so this script verifies the
defense-side numbers the paper reports, without needing BackdoorBench installed.

Layout inside the zip:
    defense_logs/<scenario>/<selection>/<defense>/[<subdir>/]<file>
      *.log / *.txt            -> metrics dict  (nc/fp/rnp/i-bau/abl/ac) or FST line
      detection_info.csv       -> TPR/FPR       (strip/scan)

Outputs (written to ../defense_results by default):
    summary.csv                 one row per scenario/selection/defense (342 rows)
    full_report.md              all scenarios, ASR/ACC/detection matrices
    by_scenario/<scenario>.md   per-scenario breakdown
    status.txt                  coverage counts

Run:  python analyze_author_defense.py                 # write outputs + print
      python analyze_author_defense.py --scenario sig_p0.03
      python analyze_author_defense.py --out-dir /tmp/x
"""

import argparse
import csv as csvmod
import io
import os
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ZIP_PATH = os.path.normpath(os.path.join(HERE, "..", "resource", "defense_logs.zip"))
DEFAULT_OUT = os.path.normpath(os.path.join(HERE, "..", "defense_results"))
sys.path.insert(0, HERE)
from parse_log import parse_metrics_text, parse_detection_text  # noqa: E402
from grid import DEFENSES, DEFENSE_ORDER, SCENARIO_MAP, SELECTIONS  # noqa: E402

DETECTION = {k for k, v in DEFENSES.items() if v[2] == "detection"}  # {strip, scan}


def _parts(name):
    """zip entry name -> (scenario, selection, defense) or None."""
    p = name.replace("\\", "/")
    seg = [s for s in p.split("/") if s]
    if len(seg) < 5 or seg[0] != "defense_logs":
        return None
    return seg[1], seg[2], seg[3]


def collect():
    """Return {(scenario, selection, defense): {'acc':,'asr':,'TPR':,'FPR':,'src':}}."""
    rows = {}
    if not os.path.exists(ZIP_PATH):
        raise SystemExit("defense_logs.zip not found at {}".format(ZIP_PATH))
    with zipfile.ZipFile(ZIP_PATH) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            key = _parts(info.filename)
            if key is None:
                continue
            scenario, selection, defense = key
            if defense not in DEFENSES:
                continue
            base = os.path.basename(info.filename)
            text = z.read(info).decode("utf-8", errors="replace")
            cell = rows.setdefault(key, {"acc": None, "asr": None,
                                         "TPR": None, "FPR": None, "src": ""})
            if base.endswith(".csv"):
                det = parse_detection_text(text)
                if det["TPR"] is not None:
                    cell["TPR"], cell["FPR"] = det["TPR"], det["FPR"]
                    cell["src"] = base
            elif base.endswith((".log", ".txt")):
                m = parse_metrics_text(text)
                if m["test_asr"] is not None:
                    cell["acc"], cell["asr"] = m["test_acc"], m["test_asr"]
                    cell["src"] = base
    return rows


# --------------------------------------------------------------------------- #
# Rendering (returns strings so the same code feeds stdout + markdown files)
# --------------------------------------------------------------------------- #
def _matrix_str(rows, scenario, field, title):
    sels = [s for s in SELECTIONS if any((scenario, s, d) in rows for d in DEFENSE_ORDER)]
    out = ["### {}  ({})\n".format(title, scenario), ""]
    hdr = "{:<10}".format("sel\\def") + "".join("{:>8}".format(d) for d in DEFENSE_ORDER)
    out += [hdr, "-" * len(hdr)]
    for s in sels:
        cells = []
        for d in DEFENSE_ORDER:
            cell = rows.get((scenario, s, d))
            if not cell:
                cells.append(".")
                continue
            v = cell.get(field)
            cells.append("-" if v is None else "{:.1f}".format(v * 100))
        out.append("{:<10}".format(s) + "".join("{:>8}".format(c) for c in cells))
    return "\n".join(out)


def _detection_str(rows, scenario):
    sels = [s for s in SELECTIONS if any((scenario, s, d) in rows for d in DETECTION)]
    if not sels:
        return ""
    out = ["### Detection TPR%/FPR%  ({})\n".format(scenario), ""]
    hdr = "{:<10}".format("sel\\def") + "".join("{:>12}".format(d) for d in ["strip", "scan"])
    out += [hdr, "-" * len(hdr)]
    for s in sels:
        cells = []
        for d in ["strip", "scan"]:
            cell = rows.get((scenario, s, d))
            cells.append("{:.0f}/{:.0f}".format(cell["TPR"] * 100, cell["FPR"] * 100)
                         if cell and cell.get("TPR") is not None else "-")
        out.append("{:<10}".format(s) + "".join("{:>12}".format(c) for c in cells))
    return "\n".join(out)


def scenario_block(rows, scenario):
    desc = SCENARIO_MAP.get(scenario, (None, None, ""))[2]
    parts = ["## {}   ({})\n".format(scenario, desc), "",
             _matrix_str(rows, scenario, "asr", "ASR after defense (%)"), "",
             _matrix_str(rows, scenario, "acc", "Clean ACC after defense (%)"), ""]
    det = _detection_str(rows, scenario)
    if det:
        parts += [det, ""]
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Output writers
# --------------------------------------------------------------------------- #
def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csvmod.writer(f)
        w.writerow(["scenario", "selection", "defense", "category",
                    "test_acc", "test_asr", "TPR", "FPR", "source_file"])
        for (scn, sel, defense), cell in sorted(rows.items()):
            w.writerow([scn, sel, defense, DEFENSES[defense][1],
                        cell.get("acc"), cell.get("asr"),
                        cell.get("TPR"), cell.get("FPR"), cell.get("src")])


def coverage_str(rows):
    out = ["## Coverage (parsed / total logs)\n"]
    for d in DEFENSE_ORDER:
        rel = {k: v for k, v in rows.items() if k[2] == d}
        if d in DETECTION:
            ok = sum(1 for v in rel.values() if v.get("TPR") is not None)
        else:
            ok = sum(1 for v in rel.values() if v.get("asr") is not None)
        out.append("- **{}**: {} / {} parsed".format(d, ok, len(rel)))
    return "\n".join(out)


def write_outputs(rows, out_dir, scenarios):
    os.makedirs(os.path.join(out_dir, "by_scenario"), exist_ok=True)
    write_csv(rows, os.path.join(out_dir, "summary.csv"))
    with open(os.path.join(out_dir, "full_report.md"), "w") as f:
        f.write("# Defense results — full report\n\n")
        f.write("> Auto-generated by `tests/analyze_author_defense.py`. ")
        f.write("Source: `resource/defense_logs.zip`. Cells = POST-DEFENSE.\n\n")
        for scn in scenarios:
            f.write(scenario_block(rows, scn) + "\n\n----\n\n")
        f.write(coverage_str(rows) + "\n")
    for scn in scenarios:
        with open(os.path.join(out_dir, "by_scenario", scn + ".md"), "w") as f:
            f.write("# {} — defense breakdown\n\n".format(scn))
            f.write(scenario_block(rows, scn) + "\n\n")
            f.write(coverage_str({k: v for k, v in rows.items()
                                  if k[0] == scn}) + "\n")
    with open(os.path.join(out_dir, "status.txt"), "w") as f:
        f.write(coverage_str(rows) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default=None, help="print only this scenario")
    ap.add_argument("--out-dir", default=DEFAULT_OUT, help="where to write outputs")
    args = ap.parse_args()

    rows = collect()
    all_scns = [s for s, _ in SCENARIO_MAP.items() if s in {k[0] for k in rows}]
    scenarios = [args.scenario] if args.scenario else all_scns
    scenarios = [s for s in scenarios if s in all_scns]

    if args.scenario:
        print(scenario_block(rows, args.scenario))
    else:
        print("=" * 70)
        print(" Author defense-log summary  (resource/defense_logs.zip)")
        print(" POST-DEFENSE values; ASR lower = defense more effective.")
        print("=" * 70)
        for scn in scenarios:
            print("\n" + scenario_block(rows, scn))
    print("\n" + coverage_str(rows))

    write_outputs(rows, args.out_dir, all_scns)
    print("\n[written] {}  (summary.csv, full_report.md, by_scenario/*, status.txt)"
          .format(args.out_dir))


if __name__ == "__main__":
    main()

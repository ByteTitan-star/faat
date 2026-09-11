"""Parse BackdoorBench defense output.

BackdoorBench defenses emit results in two shapes:

1. metrics defenses (nc/fp/fst/rnp/i-bau/abl/ac) print, at the end of
   training, a dict like::

       [INFO] [trainer_cls.py:65] {'bd_test_loss_avg_over_batch': ...,
        'test_acc': 0.9161,
        'test_asr': 0.6155,
        'test_ra': 0.2936, ...}

   We take the LAST 'test_acc' / 'test_asr' seen in the file (= final model).

2. detection defenses (strip/scan) write a detection_info.csv::

       record,TN,FP,FN,TP,TPR,FPR,target
       sig_cifar10_cleanlabel_0.03_random,932,68,1583,7417,0.8241,0.068,None

   We take the last data row's TPR / FPR.
"""

import csv
import io
import os
import re

# Matches  'test_acc': 0.9161   and   "test_acc": 0.9161   (also 0.9161,)
_METRIC_RE = {
    "test_acc": re.compile(r"['\"]test_acc['\"]\s*:\s*([0-9eE.+-]+)"),
    "test_asr": re.compile(r"['\"]test_asr['\"]\s*:\s*([0-9eE.+-]+)"),
    "test_ra":  re.compile(r"['\"]test_ra['\"]\s*:\s*([0-9eE.+-]+)"),
}

# FST (fine-tuning) uses a different line format:  Clean ACC: 0.9243 | ASR: 0.512
_FST_RE = re.compile(r"Clean ACC:\s*([0-9.]+)\s*\|\s*ASR:\s*([0-9.]+)")


def _last_float(text, regex):
    m = None
    for m in re.finditer(regex, text):
        pass
    if m is None:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def parse_metrics_text(text):
    """Return {'test_acc':, 'test_asr':, 'test_ra':} from log text (see module docstring)."""
    out = {"test_acc": None, "test_asr": None, "test_ra": None}
    for key, regex in _METRIC_RE.items():
        out[key] = _last_float(text, regex)
    # FST fallback (only fills fields the dict style did not find).
    last_fst = None
    for m in _FST_RE.finditer(text):
        last_fst = m
    if last_fst is not None:
        if out["test_acc"] is None:
            out["test_acc"] = float(last_fst.group(1))
        if out["test_asr"] is None:
            out["test_asr"] = float(last_fst.group(2))
    return out


def parse_metrics_log(path):
    """parse_metrics_text, reading from a file path."""
    if not os.path.exists(path):
        return {"test_acc": None, "test_asr": None, "test_ra": None}
    try:
        with open(path, "r", errors="replace") as f:
            return parse_metrics_text(f.read())
    except OSError:
        return {"test_acc": None, "test_asr": None, "test_ra": None}


def parse_detection_text(text):
    """Return {'TPR':, 'FPR':} from the last data row of a detection_info.csv text."""
    out = {"TPR": None, "FPR": None}
    rows = list(csv.DictReader(io.StringIO(text)))
    if not rows:
        return out
    last = rows[-1]
    for key in out:
        try:
            out[key] = float(last.get(key))
        except (TypeError, ValueError):
            out[key] = None
    return out


def parse_detection_csv(path):
    """parse_detection_text, reading from a file path."""
    if not os.path.exists(path):
        return {"TPR": None, "FPR": None}
    try:
        with open(path, "r", errors="replace") as f:
            return parse_detection_text(f.read())
    except OSError:
        return {"TPR": None, "FPR": None}


def result_kind_for(defense):
    """Lookup the parse shape for a defense key (imports grid lazily)."""
    from grid import DEFENSES
    if defense not in DEFENSES:
        return "metrics"
    return DEFENSES[defense][2]


def summarize(job):
    """Given a job dict (see worker.py / launcher.py), return a flat metrics dict.

    Picks the right parser based on the defense's result_kind. Keys always
    present: test_acc, test_asr, test_ra, TPR, FPR (None when not applicable).
    """
    defense = job["defense"]
    kind = result_kind_for(defense)
    flat = {"test_acc": None, "test_asr": None, "test_ra": None,
            "TPR": None, "FPR": None}
    if kind == "detection":
        flat.update(parse_detection_csv(job.get("detection_csv")))
    else:
        flat.update(parse_metrics_log(job.get("log")))
    return flat


def fmt(x, ndigits=4):
    return "{:.{}f}".format(x, ndigits) if isinstance(x, (int, float)) else "-"

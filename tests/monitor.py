"""Live monitor / result poller for submitted defense jobs.

Scans jobs/*.json, reads each job's lifecycle status (written by worker.py)
and parses its log / detection_info.csv for metrics, then prints one table.
With --watch it refreshes every N seconds until interrupted.

  python monitor.py            # print once
  python monitor.py --watch    # refresh every 10s
  python monitor.py --watch --interval 5
"""

import glob
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
JOBS_DIR = os.path.join(HERE, "jobs")
sys.path.insert(0, HERE)
from parse_log import summarize, fmt  # noqa: E402
from grid import DEFENSE_ORDER  # noqa: E402

STATUS_GLYPH = {
    "pending":  "·",
    "running":  "▶",
    "done":     "✓",
    "failed":   "✗",
}


def _read_job(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        # Worker writes atomically; a torn read shouldn't happen, but guard anyway.
        return None


def _elapsed(start, end):
    if not start:
        return ""
    fmt_in = "%Y-%m-%d %H:%M:%S"
    try:
        t0 = time.mktime(time.strptime(start, fmt_in))
        t1 = time.mktime(time.strptime(end, fmt_in)) if end else time.time()
        secs = int(t1 - t0)
        if secs < 60:
            return "{}s".format(secs)
        return "{}m{}s".format(secs // 60, secs % 60)
    except ValueError:
        return ""


def collect():
    jobs = []
    for path in sorted(glob.glob(os.path.join(JOBS_DIR, "*.json"))):
        job = _read_job(path)
        if job is None:
            continue
        metrics = summarize(job)
        jobs.append((job, metrics))
    # Order: by scenario, then selection, then the canonical defense order.
    def rank(j):
        job = j[0]
        try:
            d = DEFENSE_ORDER.index(job["defense"])
        except ValueError:
            d = 99
        return (job.get("scenario", ""), job.get("selection", ""), d)
    jobs.sort(key=rank)
    return jobs


def render(jobs):
    now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    counts = {"pending": 0, "running": 0, "done": 0, "failed": 0}
    for job, _ in jobs:
        counts[job.get("status", "pending")] = counts.get(job.get("status", "pending"), 0) + 1

    header = "Backdoor-defense monitor  (updated {})   total: {total}   done: {done}   running: {running}   failed: {failed}".format(
        now, total=len(jobs), **counts)
    lines = [header, "-" * len(header)]
    lines.append("{:<2}{:<14} {:<10} {:<7} {:<7} {:<7} {:<7} {:<7} {:>7}  {}".format(
        " ", "scenario", "selection", "defense", "status", "acc", "asr", "TPR/FPR", "elapsed", "log"))
    lines.append("-" * len(header))
    for job, m in jobs:
        status = job.get("status", "pending")
        glyph = STATUS_GLYPH.get(status, "?")
        if m["TPR"] is not None:
            tpr_fpr = "{}/{}".format(fmt(m["TPR"], 3), fmt(m["FPR"], 3))
            acc = asr = "-"
        else:
            tpr_fpr = "-"
            acc = fmt(m["test_acc"])
            asr = fmt(m["test_asr"])
        logname = os.path.basename(job.get("log", ""))
        lines.append("{:<2}{:<14} {:<10} {:<7} {:<7} {:<7} {:<7} {:<7} {:>7}  {}".format(
            glyph,
            str(job.get("scenario", ""))[:14],
            str(job.get("selection", ""))[:10],
            str(job.get("defense", ""))[:7],
            status[:7],
            acc, asr, tpr_fpr,
            _elapsed(job.get("start"), job.get("end")),
            logname))
    return "\n".join(lines)


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Monitor backdoor-defense jobs")
    ap.add_argument("--watch", action="store_true", help="refresh until Ctrl-C")
    ap.add_argument("--interval", type=float, default=10.0)
    args = ap.parse_args()

    if not args.watch:
        print(render(collect()))
        return
    try:
        while True:
            os.write(1, b"\x1b[2J\x1b[H")        # clear screen + cursor home
            print(render(collect()))
            print("\n(refreshing every {}s, Ctrl-C to stop)".format(args.interval))
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()

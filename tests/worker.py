"""Background job executor.

Launched detached by launcher.py, one process per job. Reads a job json,
marks the job running, executes the command (real BackdoorBench command, or a
MOCK that synthesizes a realistic log so the monitor can be validated before
BackdoorBench is installed), then marks the job done/failed.

A "job json" looks like::

    {
      "scenario": "sig_p0.03",
      "selection": "res_square",
      "defense": "fp",
      "cmd": "MOCK"  |  "python -u /path/BackdoorBench/defense/fp.py ...",
      "log": "jobs/sig_p0.03__res_square__fp.log",
      "detection_csv": "jobs/sig_p0.03__res_square__fp_detection_info.csv",
      "gpu": 1,
      "pid": <filled here>,
      "status": "pending" | "running" | "done" | "failed",
      "start": "...", "end": "...", "exit_code": <int>
    }

The status file IS the job json — the monitor reads the same file. Updates are
written atomically (write tmp -> rename) so the monitor never sees a torn read.
"""

import json
import os
import subprocess
import sys
import time

# Realistic mock values, taken from resource/defense_logs.zip (sig_p0.03/random).
# They make the monitor's table look believable during skeleton validation.
_MOCK_METRICS = {
    "nc":    {"test_acc": 0.9089, "test_asr": 0.0513},
    "fp":    {"test_acc": 0.9161, "test_asr": 0.6155},
    "fst":   {"test_acc": 0.9302, "test_asr": 0.3011},
    "rnp":   {"test_acc": 0.6345, "test_asr": 0.0000},
    "i-bau": {"test_acc": 0.8833, "test_asr": 0.0891},
    "abl":   {"test_acc": 0.4185, "test_asr": 0.0144},
    "ac":    {"test_acc": 0.8944, "test_asr": 0.9353},
}
_MOCK_DETECTION = {
    "strip": {"TPR": 0.8241, "FPR": 0.0680},
    "scan":  {"TPR": 0.9800, "FPR": 0.0000},
}


def _now():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def _atomic_write(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def _mock_run(job):
    """Synthesize a realistic BackdoorBench-style log + (for detection) a CSV.

    Sleeps a little so the monitor can actually observe a 'running' window.
    """
    defense = job["defense"]
    record = "{}_cifar10_cleanlabel_{}_{}".format(
        job["scenario"].rsplit("_", 1)[0],        # e.g. sig_p0.03 -> sig
        job["scenario"].split("_p")[-1],          # poison rate
        job["selection"])
    record = job["scenario"]                       # keep it simple/stable

    kind = "detection" if defense in _MOCK_DETECTION else "metrics"
    for step in range(3):
        time.sleep(2)
        with open(job["log"], "a") as f:
            f.write("[INFO] [mock.py] {} epoch={} still training...\n".format(_now(), step))

    if kind == "detection":
        tpr = _MOCK_DETECTION[defense]["TPR"]
        fpr = _MOCK_DETECTION[defense]["FPR"]
        tn, fp, fn, tp = 48500, int(fpr * 49300), 30, int(tpr * 1500)
        with open(job["detection_csv"], "w") as f:
            f.write("record,TN,FP,FN,TP,TPR,FPR,target\n")
            f.write("{},{},{},{},{},{},{},None\n".format(record, tn, fp, fn, tp, tpr, fpr))
    else:
        m = _MOCK_METRICS[defense]
        with open(job["log"], "a") as f:
            f.write("[INFO] [trainer_cls.py:65] {'bd_test_loss_avg_over_batch': 0.42,\n")
            f.write(" 'clean_test_loss_avg_over_batch': 0.31,\n")
            f.write(" 'test_acc': {},\n".format(m["test_acc"]))
            f.write(" 'test_asr': {},\n".format(m["test_asr"]))
            f.write(" 'test_ra': 0.05}\n")
    return 0


def run(jobfile):
    with open(jobfile) as f:
        job = json.load(f)

    job["pid"] = os.getpid()
    job["status"] = "running"
    job["start"] = _now()
    _atomic_write(jobfile, job)

    cmd = job.get("cmd", "MOCK")
    try:
        if cmd == "MOCK" or not cmd.strip():
            code = _mock_run(job)
        else:
            env = dict(os.environ)
            if job.get("gpu") is not None:
                env["CUDA_VISIBLE_DEVICES"] = str(job["gpu"])
            with open(job["log"], "a") as f:
                f.write("[worker] CMD: {}\n".format(cmd))
                f.write("[worker] start {}\n".format(_now()))
            proc = subprocess.run(cmd, shell=True, env=env,
                                  stdout=open(job["log"], "a"),
                                  stderr=subprocess.STDOUT)
            code = proc.returncode
    except Exception as e:  # noqa: BLE001 - we want every failure recorded
        with open(job["log"], "a") as f:
            f.write("[worker] EXCEPTION: {}\n".format(e))
        code = -1

    job["status"] = "done" if code == 0 else "failed"
    job["exit_code"] = code
    job["end"] = _now()
    _atomic_write(jobfile, job)
    return 0 if code == 0 else 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python worker.py <job.json>", file=sys.stderr)
        sys.exit(2)
    sys.exit(run(sys.argv[1]))

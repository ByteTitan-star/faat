"""CLI to submit backdoor-defense jobs.

Examples
--------
# Show the full grid (what would be run):
python launcher.py list

# Validate the harness end-to-end BEFORE BackdoorBench is installed:
python launcher.py run --scenario sig_p0.03 --selection res_square --defense fp --mock
python launcher.py run-all --scenario sig_p0.03 --selection res_square --mock --gpu 1
python monitor.py --watch            # live table

# Real run (once BackdoorBench is cloned + an attacked model exists):
python launcher.py run --scenario sig_p0.03 --selection res_square --defense fp \
    --real --bb-root /path/BackdoorBench \
    --yaml /path/BackdoorBench/config/defense/fp/sig_cleanlabel.yaml --gpu 1

The job state is a json file under jobs/; worker.py owns the lifecycle,
monitor.py only reads. So you can launch many jobs and walk away.
"""

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
JOBS_DIR = os.path.join(HERE, "jobs")
sys.path.insert(0, HERE)
from grid import DEFENSES, DEFENSE_ORDER, SCENARIO_MAP, defenses  # noqa: E402


def job_paths(job_id):
    base = os.path.join(JOBS_DIR, job_id)
    return {
        "json": base + ".json",
        "log": base + ".log",
        "detection_csv": base + "_detection_info.csv",
    }


def build_real_cmd(defense, bb_root, yaml_path, scenario):
    """Construct a BackdoorBench defense invocation.

    NOTE: the exact flags per defense come from BackdoorBench's own CLI and the
    yaml. The scenario name (e.g. sig_cifar10_cleanlabel_0.03_res_square) must
    match the folder produced by the ATTACK stage (the attacked model). Confirm
    the yaml path against your BackdoorBench checkout before relying on this.
    """
    py = "python -u"
    if defense == "fst":
        script = "defense/ft.py"      # FST is the 'fst' mode of FT in BackdoorBench
    else:
        script = "defense/{}.py".format(defense)
    return "cd {bb} && {py} {script} --yaml_path {yaml}".format(
        bb=bb_root, py=py, script=script, yaml=yaml_path)


def make_job(scenario, selection, defense, mode, bb_root, yaml_path, gpu):
    if scenario not in SCENARIO_MAP:
        raise SystemExit("unknown scenario '{}'; valid: {}".format(
            scenario, ", ".join(SCENARIO_MAP)))
    if defense not in DEFENSES:
        raise SystemExit("unknown defense '{}'; valid: {}".format(
            defense, ", ".join(DEFENSE_ORDER)))

    jid = "{}__{}__{}".format(scenario, selection, defense)
    p = job_paths(jid)

    if mode == "mock":
        cmd = "MOCK"
    else:
        if not bb_root or not yaml_path:
            raise SystemExit("--real requires --bb-root and --yaml")
        cmd = build_real_cmd(defense, bb_root, yaml_path, scenario)

    job = {
        "scenario": scenario,
        "selection": selection,
        "defense": defense,
        "cmd": cmd,
        "log": p["log"],
        "detection_csv": p["detection_csv"],
        "gpu": gpu,
        "pid": None,
        "status": "pending",
        "start": None,
        "end": None,
        "exit_code": None,
    }
    return jid, p, job


def submit(job, jid, p, dry_run):
    os.makedirs(JOBS_DIR, exist_ok=True)
    if dry_run:
        print("[dry-run] {} -> {}".format(jid, job["cmd"]))
        return
    with open(p["json"], "w") as f:
        json.dump(job, f, indent=2)
    # Detached: worker survives this shell. Its own stdout/stderr are unused
    # (it writes into the job log), so send them to DEVNULL.
    devnull = subprocess.DEVNULL
    subprocess.Popen(
        [sys.executable, os.path.join(HERE, "worker.py"), p["json"]],
        stdout=devnull, stderr=devnull, stdin=devnull,
        close_fds=True, start_new_session=True)
    print("[submit] {}  (status file: {})".format(jid, p["json"]))


def cmd_list(args):
    print("{:<16} {:<10} {:<8} {:<28} {:<14} {}".format(
        "scenario", "selection", "defense", "full name", "category", "result"))
    print("-" * 100)
    for folder, (attack, rate, desc) in SCENARIO_MAP.items():
        print("# {}  ({})".format(folder, desc))
        for key, full, cat, kind in defenses():
            print("{:<16} {:<10} {:<8} {:<28} {:<14} {}".format(
                "", "", key, full, cat, kind))
        print()


def cmd_run(args):
    jid, p, job = make_job(args.scenario, args.selection, args.defense,
                           args.mode, args.bb_root, args.yaml, args.gpu)
    submit(job, jid, p, args.dry_run)


def cmd_run_all(args):
    """Run every defense for one (scenario, selection)."""
    defs = DEFENSE_ORDER
    if args.defense:                       # allow restricting to a subset
        defs = [d for d in defs if d in args.defense]
    submitted = 0
    for key in defs:
        jid, p, job = make_job(args.scenario, args.selection, key,
                               args.mode, args.bb_root, args.yaml, args.gpu)
        # In real mode, each defense needs its own yaml; allow a per-defense
        # naming convention if --yaml is a directory, else reuse the one given.
        if args.mode != "mock" and args.yaml and os.path.isdir(args.yaml):
            job["cmd"] = build_real_cmd(
                key, args.bb_root,
                os.path.join(args.yaml, "{}.yaml".format(key)),
                args.scenario)
            job["yaml"] = os.path.join(args.yaml, "{}.yaml".format(key))
        submit(job, jid, p, args.dry_run)
        submitted += 1
    print("[run-all] submitted {} jobs for {}/{}".format(
        submitted, args.scenario, args.selection))


def add_mode(p):
    g = p.add_mutually_exclusive_group()
    g.add_argument("--mock", action="store_const", dest="mode", const="mock")
    g.add_argument("--real", action="store_const", dest="mode", const="real")
    p.set_defaults(mode="mock")


def main():
    ap = argparse.ArgumentParser(description="Backdoor-defense job launcher")
    sub = ap.add_subparsers(dest="cmd")
    sub.required = True

    pl = sub.add_parser("list", help="print the experiment grid")
    pl.set_defaults(func=cmd_list)

    pr = sub.add_parser("run", help="submit one job")
    pr.add_argument("--scenario", required=True)
    pr.add_argument("--selection", required=True)
    pr.add_argument("--defense", required=True)
    pr.add_argument("--gpu", type=int, default=None)
    pr.add_argument("--bb-root", default=None)
    pr.add_argument("--yaml", default=None)
    pr.add_argument("--dry-run", action="store_true")
    add_mode(pr)
    pr.set_defaults(func=cmd_run)

    pa = sub.add_parser("run-all", help="submit every defense for one scenario/selection")
    pa.add_argument("--scenario", required=True)
    pa.add_argument("--selection", required=True)
    pa.add_argument("--defense", nargs="*", default=None, help="subset of defenses")
    pa.add_argument("--gpu", type=int, default=None)
    pa.add_argument("--bb-root", default=None)
    pa.add_argument("--yaml", default=None, help="yaml file, or dir of <def>.yaml")
    pa.add_argument("--dry-run", action="store_true")
    add_mode(pa)
    pa.set_defaults(func=cmd_run_all)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

"""Experiment grid for the backdoor-defense reproduction.

Encodes the full grid the paper evaluated:
    7 attack scenarios  x  6 sample-selection strategies  x  9 defenses

Everything here is plain data so launcher.py / monitor.py can import it
without side effects. Defense metadata records, for each defense, how its
result should be parsed (BackdoorBench writes results in two different
shapes — see parse_log.py).
"""

# ---------------------------------------------------------------------------
# Defenses (the 9 from BackdoorBench that the paper evaluated against)
#   result_kind:
#     "metrics"   -> final test_acc / test_asr printed in the .log
#     "detection" -> a detection_info.csv with TPR / FPR (STRIP, SCAn)
# ---------------------------------------------------------------------------
DEFENSES = {
    "abl":   ("Anti-Backdoor Learning",                    "model_repair",  "metrics"),
    "ac":    ("Activation Clustering",                     "sample_filter", "metrics"),
    "fp":    ("Fine-Pruning",                              "model_repair",  "metrics"),
    "fst":   ("Fine-Tuning (fst mode)",                    "model_repair",  "metrics"),
    "i-bau": ("Iterative Backdoor Adversarial Unlearning", "model_repair",  "metrics"),
    "nc":    ("Neural Cleanse",                            "model_repair",  "metrics"),
    "rnp":   ("Reverse-engineering Neural Pruning",        "model_repair",  "metrics"),
    "scan":  ("Spectral Clustering Analysis (SCAn)",       "sample_filter", "detection"),
    "strip": ("STRIP (STRong Intentional Perturbation)",   "sample_filter", "detection"),
}

# Stable display order: sample-filtering defenses first, then model-repair.
DEFENSE_ORDER = ["strip", "scan", "ac", "abl", "fp", "fst", "nc", "rnp", "i-bau"]

# ---------------------------------------------------------------------------
# Sample-selection strategies (the attack-side variable; README of the repo)
#   "stealth" == Component B; it is only meaningful for blend / badnet.
# ---------------------------------------------------------------------------
SELECTIONS = ["random", "forget", "res_log", "res_linear", "res_square", "stealth"]
STEALTH_VALID = {"blend", "blend_C", "badnet", "badnet_C"}  # scenarios where stealth applies

# ---------------------------------------------------------------------------
# Attack scenarios (matches the folder names inside resource/defense_logs.zip)
#   (folder, attack_key, poison_rate, description)
# ---------------------------------------------------------------------------
SCENARIOS = [
    ("badnet_p0.03",   "badnet",   "0.03", "BadNets"),
    ("badnet_C_p0.03", "badnet_C", "0.03", "BadNets + Component C"),
    ("blend_p0.03",    "blend",    "0.03", "Blended"),
    ("blend_C_p0.03",  "blend_C",  "0.03", "Blended + Component C"),
    ("blend_p0.05",    "blend",    "0.05", "Blended (higher poison rate)"),
    ("ctrl_p0.03",     "ctrl",     "0.03", "Ctrl (clean-label)"),
    ("sig_p0.03",      "sig",      "0.03", "Signal (Sig)"),
]

SCENARIO_MAP = {folder: (attack, rate, desc) for folder, attack, rate, desc in SCENARIOS}


def selections_for(attack_key):
    """Selections valid for a given attack (stealth only where Component B applies)."""
    if attack_key in STEALTH_VALID:
        return SELECTIONS
    return [s for s in SELECTIONS if s != "stealth"]


def defenses():
    """Yield (key, full_name, category, result_kind) in display order."""
    for key in DEFENSE_ORDER:
        full, cat, kind = DEFENSES[key]
        yield key, full, cat, kind


def job_id(scenario, selection, defense):
    return "{}__{}__{}".format(scenario, selection, defense)


def all_jobs():
    """Enumerate the full grid as (scenario, attack, selection, defense) tuples.

    Matches the layout of resource/defense_logs.zip. Useful for run-all / planning.
    """
    for folder, attack, rate, desc in SCENARIOS:
        for sel in selections_for(attack):
            for key in DEFENSE_ORDER:
                yield folder, attack, sel, key

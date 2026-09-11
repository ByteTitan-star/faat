"""Defense evaluation for KST / Narcissus victim models on CIFAR-10.

Two defenses:
  * STRIP (generic behaviour defence): perturbation-consistency. Catches any
    working backdoor (KST and Narcissus alike) -- a baseline, not the differentiator.
  * Frequency-signature defence (the KST differentiator): estimate the trigger's
    spectral footprint as  mean|FFT(triggered)|^2 - mean|FFT(clean)|^2, then score
    each image by inner product with that footprint. AUC separating triggered from
    clean.  Narcissus (peaked delta) -> high AUC (detectable).  KST (flat delta)
    -> AUC ~0.5 (undetectable).  Also reports the footprint peak/median ratio.

Expects each run saved via kst_sdt_validate.py --save_model, i.e.
  results/kst_sdt/<tag>/{model_last.pth, delta.pth, args.json}

Usage:
  python faat/defense_eval.py --tags kst_eps0.031_pr0.05_s1 narcissus_eps0.031_pr0.05_s1
  python faat/defense_eval.py --tags kst_eps0.063_pr0.05_s1 --device cuda:0
"""
import argparse, os, sys, json
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from cifar_resnet import ResNet18                       # noqa: E402
from faat.defenses import strip_scores, _auc            # noqa: E402
from faat.kst_sdt_validate import load_cifar_tensors    # noqa: E402

OUT = os.path.join(ROOT, "results", "kst_sdt")


def load_run(tag, device):
    run_dir = os.path.join(OUT, tag)
    sd = torch.load(os.path.join(run_dir, "model_last.pth"), map_location=device)
    model = ResNet18(num_classes=10).to(device)
    model.load_state_dict(sd); model.eval()
    delta = torch.load(os.path.join(run_dir, "delta.pth"), map_location=device).to(device)
    args = json.load(open(os.path.join(run_dir, "args.json")))
    return model, delta, args


@torch.no_grad()
def frequency_signature_auc(x_clean, x_triggered, device, n_use=2000):
    """Spectral-signature defence AUC + footprint peak/median."""
    nc = min(n_use, len(x_clean), len(x_triggered))
    xc = x_clean[:nc]; xt = x_triggered[:nc]
    half = nc // 2

    def avg_spec(xs):
        Fd = torch.fft.rfft2(xs.to(device), norm="ortho")
        return (Fd.abs() ** 2).mean(0)                      # [3,H,W//2+1]

    sig = (avg_spec(xt[:half]) - avg_spec(xc[:half])).flatten()    # [D]
    peak_over_med = (sig.max() / sig.median().clamp(min=1e-12)).item()

    def score(xs):
        Fd = torch.fft.rfft2(xs.to(device), norm="ortho")
        m2 = (Fd.abs() ** 2).flatten(1)                      # [N,D]
        return (m2 * sig[None]).sum(1).cpu().numpy()

    sc_t = score(xt[half:]); sc_c = score(xc[half:])
    scores = np.concatenate([sc_t, sc_c])
    labels = np.array([1] * len(sc_t) + [0] * len(sc_c))
    return _auc(scores, labels), peak_over_med


def strip_auc(model, x_clean, x_triggered, device, n=500, n_perturb=50):
    """STRIP: triggered images stay confident under perturbation -> low entropy."""
    rng = np.random.RandomState(0)
    it = rng.choice(len(x_triggered), min(n, len(x_triggered)), replace=False)
    ic = rng.choice(len(x_clean), min(n, len(x_clean)), replace=False)
    ent_t = strip_scores(model, x_triggered[it], device, n_perturb)
    ent_c = strip_scores(model, x_clean[ic], device, n_perturb)
    scores = -np.concatenate([ent_t, ent_c])                # higher = backdoored
    labels = np.array([1] * len(ent_t) + [0] * len(ent_c))
    return _auc(scores, labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--target", type=int, default=0)
    args = ap.parse_args()
    device = args.device
    torch.cuda.set_device(device)

    xtr, ytr, xte, yte = load_cifar_tensors()

    rows = []
    for tag in args.tags:
        if not os.path.isdir(os.path.join(OUT, tag)):
            print(f"[skip] {tag}: no run dir"); continue
        model, delta, ra = load_run(tag, device)
        eps = ra["eps"]
        # triggered test set (universal delta for kst/narcissus)
        xte_trig = torch.clamp(xte + delta.cpu(), 0, 1)
        # also re-measure ASR/BA from the model for a self-contained row
        with torch.no_grad():
            ba = (model(xte.to(device)).argmax(1).cpu() == yte).float().mean().item()
            mask = yte != args.target
            asr = (model(xte_trig.to(device)).argmax(1).cpu() == args.target)[mask].float().mean().item()
        f_auc, f_peak = frequency_signature_auc(xte, xte_trig, device)
        s_auc = strip_auc(model, xte, xte_trig, device)
        row = {"tag": tag, "trigger": ra["trigger"], "eps": eps,
               "ASR": round(asr, 4), "BA": round(ba, 4),
               "STRIP_AUC": round(s_auc, 3),
               "FreqSig_AUC": round(f_auc, 3),
               "FreqSig_peak_over_median": round(f_peak, 2)}
        rows.append(row)
        print(f"{tag}: ASR={asr:.3f} BA={ba:.3f} | STRIP_AUC={s_auc:.3f} | "
              f"FreqSig_AUC={f_auc:.3f} peak/med={f_peak:.1f}", flush=True)
        del model
        torch.cuda.empty_cache()

    with open(os.path.join(OUT, "defense_eval.json"), "w") as f:
        json.dump(rows, f, indent=2)
    print("\n=== summary ===")
    print(f"{'trigger':<12}{'eps':<8}{'ASR':<8}{'BA':<8}{'STRIP':<8}{'FreqSig':<9}{'peak/med':<9}")
    for r in rows:
        print(f"{r['trigger']:<12}{r['eps']:<8.4f}{r['ASR']:<8.3f}{r['BA']:<8.3f}"
              f"{r['STRIP_AUC']:<8.3f}{r['FreqSig_AUC']:<9.3f}{r['FreqSig_peak_over_median']:<9.1f}")
    print(f"\nsaved -> {OUT}/defense_eval.json")


if __name__ == "__main__":
    main()

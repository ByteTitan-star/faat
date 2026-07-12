"""Validate KST (4th-order cumulant signature) and SDT (Stein-discrepancy /
score-domain) backdoor triggers on CIFAR-10.

Self-contained: builds each trigger, poisons CIFAR-10, trains ResNet-18,
evals ASR/BA/SSIM, and runs (a)/(b)/(c) diagnostics.

  * KST        : universal delta, FFT-phase parameterised (flat magnitude
                 spectrum -> no frequency peak by construction), objective is
                 the 4th-order polynomial  s(x) = Sum_m prod_k (d_m[k] . x_tilde)
                 on ZCA-whitened pixels.   -> validates (a) spectrally flat,
                 (b) noise-like not a pattern, (c) 4th-order discriminator.
  * SDT        : input-adaptive delta, maximises the Stein residual
                 S_g(x) = g(x).score(x) + div g  under a manifold-tangency
                 constraint delta _|_ score(x).  -> validates (a) score domain
                 (gradient of log-density, unused), (b) optimisation primitive
                 not pattern overlay, (c) manifold-tangent structural invariant.
  * narcissus  : loads existing noise_01000.pth, rescaled to the same L_inf
                 budget, as the frequency-localised baseline for contrast.

Usage:
  python faat/kst_sdt_validate.py --trigger kst       --gpu 0
  python faat/kst_sdt_validate.py --trigger sdt       --gpu 1
  python faat/kst_sdt_validate.py --trigger narcissus --gpu 2
  python faat/kst_sdt_validate.py --trigger kst --quick   # 1-epoch smoke test

All artifacts cached under resource/kst_sdt/ ; results under results/kst_sdt/.
"""
import argparse, os, sys, json, time, copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from cifar_resnet import ResNet18, UnetGenerator          # noqa: E402
from faat.losses import ssim                               # noqa: E402

RES = os.path.join(ROOT, "resource", "kst_sdt")
OUT = os.path.join(ROOT, "results", "kst_sdt")
os.makedirs(RES, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

CIFAR_MEAN_PERPIX = None  # filled lazily


# =========================================================================== #
# Data
# =========================================================================== #
def load_cifar_tensors():
    """Return (x_train,y_train,x_test,y_test) float tensors in [0,1], shape [N,3,32,32]."""
    train = datasets.CIFAR10(os.path.join(ROOT, "data"), train=True,  download=True,
                             transform=transforms.ToTensor())
    test  = datasets.CIFAR10(os.path.join(ROOT, "data"), train=False, download=True,
                             transform=transforms.ToTensor())
    def to_tensor(ds):
        ld = torch.utils.data.DataLoader(ds, batch_size=1000, shuffle=False, num_workers=4)
        xs, ys = [], []
        for x, y in ld:
            xs.append(x); ys.append(y)
        return torch.cat(xs), torch.cat(ys)
    xtr, ytr = to_tensor(train)
    xte, yte = to_tensor(test)
    return xtr, ytr, xte, yte


# =========================================================================== #
# ZCA whitening  (for KST 4th-order objective: makes clean projections ~Gaussian)
# =========================================================================== #
def fit_zca(x, n_fit=10000, eps=1e-3):
    """x: [N,3,32,32].  Returns (mu[3072], W[3072,3072]) on CPU float32."""
    cache = os.path.join(RES, "zca.pt")
    if os.path.exists(cache):
        return torch.load(cache)
    idx = np.random.RandomState(0).choice(len(x), min(n_fit, len(x)), replace=False)
    xf = x[idx].reshape(-1, 3072).numpy().astype(np.float64)
    mu = xf.mean(0)
    xc = xf - mu
    cov = (xc.T @ xc) / len(xf)
    w, V = np.linalg.eigh(cov)
    w = np.clip(w, 0.0, None)
    W = (V * (1.0 / np.sqrt(w + eps))) @ V.T
    mu_t = torch.tensor(mu, dtype=torch.float32)
    W_t = torch.tensor(W, dtype=torch.float32)
    torch.save((mu_t, W_t), cache)
    return mu_t, W_t


def zca_whiten(x, mu, W):
    """x: [B,3,32,32] (cuda or cpu) -> whitened [B,3,32,32]."""
    B = x.shape[0]
    xf = x.reshape(B, -1)
    mu = mu.to(xf); W = W.to(xf)
    return ((xf - mu) @ W.T).reshape(B, 3, 32, 32)


# =========================================================================== #
# Score network (denoising score matching) for SDT
# =========================================================================== #
def train_score_net(x_train, device, epochs=15, sigma=0.1, batch=256):
    cache = os.path.join(RES, f"score_net_sigma{sigma}.pt")
    net = UnetGenerator(in_channels=3, nf=32, out_channel=3).to(device)
    if os.path.exists(cache):
        net.load_state_dict(torch.load(cache, map_location=device))
        return net.eval()
    opt = torch.optim.Adam(net.parameters(), lr=2e-4)
    N = len(x_train)
    xtr = x_train.to(device)
    net.train()
    for ep in range(epochs):
        perm = torch.randperm(N, device=device)
        tot = 0.0
        for i in range(0, N, batch):
            idx = perm[i:i+batch]
            x = xtr[idx]
            eps = torch.randn_like(x) * sigma
            x_noisy = x + eps
            eps_hat = net(x_noisy)
            loss = F.mse_loss(eps_hat, eps)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        print(f"  [score-net] ep{ep} mse={tot/N:.5e}", flush=True)
    torch.save(net.state_dict(), cache)
    return net.eval()


def score_of(net, x, sigma=0.1):
    """Estimate score = -eps_hat / sigma^2.  x: [B,3,32,32]."""
    with torch.no_grad():
        eps_hat = net(x)
    return -eps_hat / (sigma ** 2)


# =========================================================================== #
# KST trigger  (universal, flat-spectrum, 4th-order)
# =========================================================================== #
class KST:
    def __init__(self, r=4, seed=1234, device="cuda"):
        g = torch.Generator().manual_seed(seed)
        self.r = r
        self.dirs = []                                   # r tensors [4,3072]
        for _ in range(r):
            d = torch.randn(4, 3072, generator=g)
            d = F.normalize(d, dim=1)
            self.dirs.append(d.to(device))
        self.device = device

    def s(self, x_whitened):
        """x_whitened: [B,3,32,32] -> s(x) [B].  4th-order polynomial."""
        B = x_whitened.shape[0]
        xf = x_whitened.reshape(B, -1)                  # [B,3072]
        total = torch.zeros(B, device=xf.device)
        for d in self.dirs:                             # d:[4,3072]
            prods = xf @ d.T                            # [B,4]
            total = total + prods.prod(dim=1)           # product of 4 projections
        return total

    def build(self, x_clean, mu, W, eps, steps=400, batch=512, device="cuda",
              proxy=None, target=0, alpha=0.0, S=500.0):
        """Optimise FFT phases (flat magnitude -> spectrally flat delta).

        Objective (KST-Learn when proxy given):
            min  CE_proxy(x+delta, target)  -  alpha * s(x+delta) / S
        s.t. flat magnitude spectrum (hard, via phase parameterisation) + Linf=eps.
        proxy=None -> original KST (pure s maximisation)."""
        use_ce = proxy is not None
        cache = os.path.join(RES, f"kst_delta_r{self.r}_eps{eps:.4f}_a{alpha}_ce{int(use_ce)}.pt")
        if os.path.exists(cache):
            return torch.load(cache, map_location=device)
        x = x_clean.to(device)
        mu = mu.to(device); W = W.to(device)
        phase = torch.zeros(3, 32, 17, device=device, requires_grad=True)
        mag = torch.ones(3, 32, 17, device=device)
        mag[:, 0, 0] = 0.0                               # kill DC -> zero-mean delta
        opt = torch.optim.Adam([phase], lr=0.1)
        idx_pool = torch.randperm(len(x), device=device)[:min(batch*4, len(x))]
        tgt = torch.full((batch,), target, dtype=torch.long, device=device)
        for step in range(steps):
            idx = idx_pool[torch.randint(0, len(idx_pool), (batch,), device=device)]
            xb = x[idx]
            Fd = mag * torch.exp(1j * phase)             # [3,32,17] complex
            delta = torch.fft.irfft2(Fd, s=(32, 32), norm="ortho")   # [3,32,32]
            dmax = delta.abs().max().clamp(min=1e-6)
            delta_n = eps * (delta / dmax)
            x_adv = torch.clamp(xb + delta_n, 0, 1)
            if use_ce:
                loss = F.cross_entropy(proxy(x_adv), tgt)
                if alpha > 0:
                    s_val = self.s(zca_whiten(x_adv, mu, W)).mean()
                    loss = loss - alpha * (s_val / S)
                extra = f"ce={loss.item():.3f}"
            else:
                s_val = self.s(zca_whiten(x_adv, mu, W)).mean()
                loss = -s_val
                extra = f"s={s_val.item():.3f}"
            opt.zero_grad(); loss.backward(); opt.step()
            if step % 50 == 0:
                print(f"  [kst] step{step} {extra} Linf={delta_n.abs().max().item():.4f}", flush=True)
        with torch.no_grad():
            Fd = mag * torch.exp(1j * phase)
            delta = torch.fft.irfft2(Fd, s=(32, 32), norm="ortho")
            dmax = delta.abs().max().clamp(min=1e-6)
            delta = eps * (delta / dmax)
            delta = delta - delta.mean()                 # exact zero mean
        torch.save(delta, cache)
        return delta

    def apply(self, x, delta):
        """Universal delta: broadcast-add to [B,3,32,32] (clamped to [0,1])."""
        return torch.clamp(x + delta.to(x.device), 0.0, 1.0)


# =========================================================================== #
# SDT trigger  (input-adaptive, Stein-residual, manifold-tangent)
# =========================================================================== #
class SDT:
    def __init__(self, seed=4321, device="cuda"):
        g = torch.Generator().manual_seed(seed)
        # per-pixel affine Stein key  g(x) = a (x - mu_global) + b   (broadcast)
        self.a = (torch.randn(3, 32, 32, generator=g) * 0.1).to(device)
        self.b = (torch.randn(3, 32, 32, generator=g) * 0.1).to(device)
        self.device = device

    def Sg(self, x, score, mu_global):
        """Stein residual variable part:  sum (a(x-mu)+b) * score.  x,score:[B,3,32,32] -> [B]."""
        g = self.a[None] * (x - mu_global.to(x)[None]) + self.b[None]      # [B,3,32,32]
        return (g * score).flatten(1).sum(1)

    def build_batch(self, x_batch, score_net, mu_global, eps, steps=30, sigma=0.1, lam_tang=0.3):
        """Per-image PGD: maximise Sg(x+delta) s.t. Linf<=eps, zero-mean,
        with a SOFT tangency penalty  lam_tang * <delta,score>^2 / ||score||^2.
        (Hard tangency kills Sg because Sg is dominated by g.score, which is
        score-aligned -- a fundamental tension we expose and report.)"""
        x = x_batch.clone().detach()
        delta = torch.zeros_like(x, requires_grad=True)
        opt = torch.optim.Adam([delta], lr=eps * 0.5)
        for step in range(steps):
            x_adv = torch.clamp(x + delta, 0, 1)
            score = score_of(score_net, x_adv, sigma)              # [B,3,32,32]
            Sg = self.Sg(x_adv, score, mu_global).mean()
            s = score.flatten(1)                                   # [B,N]
            d = delta.flatten(1)                                   # [B,N]
            tang = ((d * s).sum(1) ** 2) / (s * s).sum(1).clamp(min=1e-6)
            loss = -Sg + lam_tang * tang.mean()
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                delta.sub_(delta.mean(dim=(1, 2, 3), keepdim=True))   # zero-mean
                delta.clamp_(-eps, eps)                              # box (final, hard)
            if step % 10 == 0:
                print(f"  [sdt] step{step} Sg={Sg.item():.4f} tang={tang.mean().item():.4f} Linf={delta.abs().max().item():.4f}", flush=True)
        return delta.detach()

    def apply_to_dataset(self, x, score_net, mu_global, eps, batch=256, sigma=0.1, steps=30):
        """Build per-image delta for every image in x [N,3,32,32], return x+delta in [0,1]."""
        out = torch.empty_like(x)
        for i in range(0, len(x), batch):
            xb = x[i:i+batch].to(self.device)
            d = self.build_batch(xb, score_net, mu_global, eps, steps=steps, sigma=sigma)
            out[i:i+batch] = torch.clamp(xb + d, 0, 1).cpu()
        return out


# =========================================================================== #
# Narcissus baseline trigger (frequency-localised, for spectral contrast)
# =========================================================================== #
def build_narcissus(eps, device="cuda"):
    cache = os.path.join(RES, f"narc_delta_eps{eps:.4f}.pt")
    if os.path.exists(cache):
        return torch.load(cache, map_location=device)
    src = os.path.join(ROOT, "resource", "narcissus", "noise_01000.pth")
    noise = torch.load(src, map_location="cpu")
    if isinstance(noise, dict):
        noise = next(v for v in noise.values() if torch.is_tensor(v))
    noise = noise.float().squeeze()
    if noise.dim() == 4:
        noise = noise[0]
    assert noise.shape == (3, 32, 32), f"unexpected narcissus shape {noise.shape}"
    # rescale to Linf = eps
    noise = noise / (noise.abs().max().clamp(min=1e-6))
    noise = eps * noise
    noise = noise - noise.mean()
    delta = noise.to(device)
    torch.save(delta, cache)
    return delta


# =========================================================================== #
# Victim training
# =========================================================================== #
def train_victim(x_train, y_train, x_test, y_test, target, epochs, lr=0.1,
                 batch=256, device="cuda", seed=0, log_every=5):
    torch.manual_seed(seed)
    model = ResNet18(num_classes=10).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[int(epochs*0.5), int(epochs*0.75)], gamma=0.1)
    xtr = x_train.to(device); ytr = y_train.to(device)
    xte = x_test.to(device);  yte = y_test.to(device)
    N = len(xtr)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(N, device=device)
        tot, correct = 0.0, 0
        for i in range(0, N, batch):
            idx = perm[i:i+batch]
            out = model(xtr[idx])
            loss = F.cross_entropy(out, ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx)
            correct += (out.argmax(1) == ytr[idx]).sum().item()
        sched.step()
        if ep % log_every == 0 or ep == epochs-1:
            ba = eval_clean(model, xte, yte, device, batch)
            print(f"  [victim] ep{ep} loss={tot/N:.4f} train_acc={correct/N:.4f} BA={ba:.4f}", flush=True)
    return model


@torch.no_grad()
def eval_clean(model, x, y, device, batch=500):
    model.eval()
    correct = 0
    for i in range(0, len(x), batch):
        out = model(x[i:i+batch].to(device))
        correct += (out.argmax(1) == y[i:i+batch].to(device)).sum().item()
    return correct / len(x)


@torch.no_grad()
def eval_asr(model, x_triggered, y_true, target, device, batch=500):
    """ASR over test images whose true label != target: fraction classified as target."""
    model.eval()
    mask = y_true != target
    xt = x_triggered[mask]
    yt = y_true[mask]
    if len(xt) == 0:
        return 0.0
    as_target = 0
    for i in range(0, len(xt), batch):
        out = model(xt[i:i+batch].to(device))
        as_target += (out.argmax(1) == target).sum().item()
    return as_target / len(xt)


# =========================================================================== #
# Diagnostics for (a)/(b)/(c)
# =========================================================================== #
def spectral_peak_ratio(delta):
    """max |FFT|^2 / median |FFT|^2  (high => localised peak like Narcissus)."""
    d = delta.detach().cpu()
    if d.dim() == 4:
        d = d[0]
    Fd = torch.fft.rfft2(d, norm="ortho")
    mag2 = Fd.abs() ** 2
    mag2 = mag2.flatten()
    return (mag2.max() / mag2.median()).item(), mag2.numpy()


def on_manifold_score_norm(score_net, x_clean, x_triggered, sigma=0.1, batch=500, device="cuda"):
    """||score(x+delta)|| / ||score(x)||  (~1 on-manifold, >1 off-manifold)."""
    def mean_score_norm(xs):
        tot, n = 0.0, 0
        for i in range(0, len(xs), batch):
            sc = score_of(score_net, xs[i:i+batch].to(device), sigma)
            tot += sc.flatten(1).norm(dim=1).sum().item()
            n += len(sc)
        return tot / n
    return mean_score_norm(x_triggered) / max(mean_score_norm(x_clean), 1e-8)


def stealth_metrics(x_clean, x_triggered, batch=500):
    """SSIM (mean), L2 (mean per-sample), Linf (max)."""
    ss, l2, linf = 0.0, 0.0, 0.0
    for i in range(0, len(x_clean), batch):
        a = x_clean[i:i+batch]; b = x_triggered[i:i+batch]
        ss += ssim(a, b).item() * len(a)
        diff = (b - a).flatten(1)
        l2 += diff.norm(dim=1).sum().item()
        linf = max(linf, diff.abs().max().item())
    n = len(x_clean)
    return ss / n, l2 / n, linf


# =========================================================================== #
# Main
# =========================================================================== #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trigger", required=True, choices=["kst", "sdt", "narcissus"])
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--poison_rate", type=float, default=0.05)
    ap.add_argument("--eps", type=float, default=8/255, help="Linf budget in [0,1]")
    ap.add_argument("--target", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--save_model", action="store_true",
                    help="save victim model_last.pth + args.json for defense eval")
    ap.add_argument("--proxy_path", default=None,
                    help="clean ResNet18 proxy for KST-Learn CE term (None=original KST)")
    ap.add_argument("--kst_alpha", type=float, default=0.0,
                    help="weight on 4th-order s term in KST-Learn (0=pure CE)")
    args = ap.parse_args()

    device = f"cuda:{args.gpu}"
    torch.cuda.set_device(args.gpu)
    np.random.seed(args.seed); torch.manual_seed(args.seed)

    tag = f"{args.trigger}_eps{args.eps:.3f}_pr{args.poison_rate}_s{args.seed}"
    if args.trigger == "kst" and args.proxy_path:
        tag = tag + f"_a{args.kst_alpha}"
    if args.quick:
        tag += "_quick"
    print(f"\n===== {tag}  device={device} =====", flush=True)

    # ---- data ----
    xtr, ytr, xte, yte = load_cifar_tensors()
    mu_g = xtr.mean(0)                                  # global per-pixel mean [3,32,32]
    if args.quick:
        xtr = xtr[:2000]; ytr = ytr[:2000]; xte = xte[:500]; yte = yte[:500]
    print(f"data: train={len(xtr)} test={len(xte)}", flush=True)

    # ---- poison indices (random subset, exclude target class for clean ASR) ----
    rng = np.random.RandomState(args.seed)
    n_poison = int(args.poison_rate * len(xtr))
    cand = np.where(ytr.numpy() != args.target)[0]
    pidx = rng.choice(cand, n_poison, replace=False)
    pidx_set = set(int(i) for i in pidx)
    print(f"poison {n_poison} ({args.poison_rate*100:.1f}%) -> target={args.target}", flush=True)

    # ---- build trigger + poisoned train set ----
    t0 = time.time()
    if args.trigger == "kst":
        mu, W = fit_zca(xtr)
        kst = KST(r=4, device=device)
        proxy = None
        if args.proxy_path:
            from cifar_resnet import ResNet18 as _R18
            sd = torch.load(args.proxy_path, map_location=device)
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            proxy = _R18(num_classes=10).to(device)
            proxy.load_state_dict(sd); proxy.eval()
            for p in proxy.parameters():
                p.requires_grad_(False)
            print(f"loaded proxy {args.proxy_path} (KST-Learn, alpha={args.kst_alpha})", flush=True)
        delta = kst.build(xtr, mu, W, args.eps,
                          steps=(1000 if proxy else 400) if not args.quick else 60, device=device,
                          proxy=proxy, target=args.target, alpha=args.kst_alpha)
        delta_cpu = delta.cpu()
        xtr_p = xtr.clone(); xtr_p[pidx] = torch.clamp(xtr[pidx] + delta_cpu, 0, 1)
        ytr_p = ytr.clone(); ytr_p[pidx] = args.target
        # test-triggered (universal delta)
        xte_trig = torch.clamp(xte + delta_cpu, 0, 1)
        # diagnostic stats handle
        diag = {"delta_universal": delta_cpu}
    elif args.trigger == "narcissus":
        delta = build_narcissus(args.eps, device=device).cpu()
        xtr_p = xtr.clone(); xtr_p[pidx] = torch.clamp(xtr[pidx] + delta, 0, 1)
        ytr_p = ytr.clone(); ytr_p[pidx] = args.target
        xte_trig = torch.clamp(xte + delta, 0, 1)
        diag = {"delta_universal": delta}
    else:  # sdt
        score_net = train_score_net(xtr, device, epochs=15 if not args.quick else 3, sigma=0.1)
        sdt = SDT(device=device)
        mu_gd = mu_g.to(device)
        # build per-image delta for poison set (train) and ALL test images
        print("building SDT deltas (train poison)...", flush=True)
        d_train = sdt.apply_to_dataset(xtr[pidx], score_net, mu_gd, args.eps,
                                       steps=30 if not args.quick else 8)
        xtr_p = xtr.clone(); xtr_p[pidx] = d_train
        ytr_p = ytr.clone(); ytr_p[pidx] = args.target
        print("building SDT deltas (test)...", flush=True)
        xte_trig = sdt.apply_to_dataset(xte, score_net, mu_gd, args.eps,
                                        steps=30 if not args.quick else 8)
        diag = {"score_net": score_net, "sdt": sdt, "mu_g": mu_g}
    print(f"trigger built in {time.time()-t0:.1f}s", flush=True)

    # ---- train victim ----
    model = train_victim(xtr_p, ytr_p, xte, yte, args.target,
                         epochs=(args.epochs if not args.quick else 1),
                         device=device, seed=args.seed, log_every=5)

    # ---- eval ----
    ba = eval_clean(model, xte, yte, device)
    asr = eval_asr(model, xte_trig, yte, args.target, device)
    ssim_v, l2_v, linf_v = stealth_metrics(xte, xte_trig)
    print(f"\n===== RESULT {tag} =====", flush=True)
    print(f"ASR={asr:.4f}  BA={ba:.4f}  SSIM={ssim_v:.4f}  L2={l2_v:.4f}  Linf={linf_v:.4f}", flush=True)

    # ---- diagnostics (a)/(b)/(c) ----
    diag_out = {"trigger": args.trigger, "ASR": asr, "BA": ba, "SSIM": ssim_v,
                "L2": l2_v, "Linf": linf_v, "eps": args.eps, "poison_rate": args.poison_rate}

    if "delta_universal" in diag:
        d = diag["delta_universal"]
        peak, mag2 = spectral_peak_ratio(d)
        diag_out["spectral_peak_over_median"] = peak
        print(f"(a) spectral peak/median = {peak:.3f}  (Narcissus~high, KST~1-2)", flush=True)
        # save delta visualisation + spectrum
        np.save(os.path.join(OUT, f"{tag}_fft_mag2.npy"), mag2)
        save_delta_png(d, os.path.join(OUT, f"{tag}_delta.png"))
    else:  # sdt
        score_net = diag["score_net"]; sdt = diag["sdt"]
        # (a) on-manifold: score norm ratio
        # use a subset of test for speed
        sub = min(1000, len(xte))
        ratio = on_manifold_score_norm(score_net, xte[:sub], xte_trig[:sub], device=device)
        diag_out["score_norm_ratio_triggered_over_clean"] = ratio
        print(f"(a/c) ||score(x+delta)||/||score(x)|| = {ratio:.3f}  (~1 on-manifold, >1 off)", flush=True)
        # (c) Stein-residual separation
        with torch.no_grad():
            sc_clean = score_of(score_net, xte[:sub].to(device), 0.1)
            sc_trig = score_of(score_net, xte_trig[:sub].to(device), 0.1)
            Sg_clean = sdt.Sg(xte[:sub].to(device), sc_clean, diag["mu_g"].to(device)).cpu().numpy()
            Sg_trig = sdt.Sg(xte_trig[:sub].to(device), sc_trig, diag["mu_g"].to(device)).cpu().numpy()
        sep = abs(Sg_trig.mean() - Sg_clean.mean()) / max(np.sqrt(Sg_clean.var()+Sg_trig.var()), 1e-8)
        diag_out["Sg_dprime"] = float(sep)
        diag_out["Sg_clean_mean"] = float(Sg_clean.mean()); diag_out["Sg_trig_mean"] = float(Sg_trig.mean())
        print(f"(c) S_g d-prime (triggered vs clean) = {sep:.3f}", flush=True)
        np.save(os.path.join(OUT, f"{tag}_Sg_clean.npy"), Sg_clean)
        np.save(os.path.join(OUT, f"{tag}_Sg_trig.npy"), Sg_trig)

    # for KST also report s(x) separation (the 4th-order discriminator)
    if args.trigger == "kst":
        mu, W = fit_zca(xtr)
        kst = KST(r=4, device=device)
        sub = min(2000, len(xte))
        with torch.no_grad():
            s_clean = kst.s(zca_whiten(xte[:sub], mu, W).to(device)).cpu().numpy()
            s_trig = kst.s(zca_whiten(xte_trig[:sub], mu, W).to(device)).cpu().numpy()
        sep = abs(s_trig.mean() - s_clean.mean()) / max(np.sqrt(s_clean.var()+s_trig.var()), 1e-8)
        diag_out["s_dprime"] = float(sep)
        diag_out["s_clean_mean"] = float(s_clean.mean()); diag_out["s_trig_mean"] = float(s_trig.mean())
        print(f"(c) s(x) d-prime (triggered vs clean) = {sep:.3f}", flush=True)
        np.save(os.path.join(OUT, f"{tag}_s_clean.npy"), s_clean)
        np.save(os.path.join(OUT, f"{tag}_s_trig.npy"), s_trig)

    with open(os.path.join(OUT, f"{tag}_result.json"), "w") as f:
        json.dump(diag_out, f, indent=2)
    print(f"saved -> {OUT}/{tag}_result.json", flush=True)

    if args.save_model:
        run_dir = os.path.join(OUT, tag)
        os.makedirs(run_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(run_dir, "model_last.pth"))
        with open(os.path.join(run_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
        # also copy the universal delta (kst/narcissus) for self-contained defense eval
        if "delta_universal" in diag:
            torch.save(diag["delta_universal"], os.path.join(run_dir, "delta.pth"))
        print(f"model+args saved -> {run_dir}/", flush=True)


def save_delta_png(delta, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        d = delta.detach().cpu().numpy()
        if d.ndim == 4:
            d = d[0]
        d = np.transpose(d, (1, 2, 0))
        d = (d - d.min()) / (d.max() - d.min() + 1e-8)
        fig, ax = plt.subplots(1, 1, figsize=(2, 2))
        ax.imshow(d); ax.axis("off")
        plt.savefig(path, bbox_inches="tight", pad_inches=0); plt.close()
    except Exception as e:
        print(f"  (skip delta png: {e})", flush=True)


if __name__ == "__main__":
    main()

"""ChromaTrigger: a clean-label backdoor trigger realised as an INVISIBLE colour-space
remap (per-channel piecewise-linear LUT) plus a smooth LOW-FREQUENCY spectral pattern.

Why this is NOT Narcissus / Badnets / Blended / MultiBpp:
  * Narcissus  : free-form per-pixel additive noise (delta [3,H,W], ~3000 free DOF).
  * Badnets/Blend: a visible patch / full-image blend.
  * MultiBpp   : DESTRUCTIVE bit-plane quantisation of RGB channels (visible banding).
  * ChromaTrigger: the trigger is a SMOOTH, GLOBAL colour transform gamma (per-channel
    piecewise-linear LUT, near-identity) + a LOW-FREQUENCY pattern P (weighted sum of
    the lowest 2D cosine modes). x' = clamp( gamma(x) + P , 0, 1).

Properties that follow from the parameterisation (not from a loss term):
  * INVISIBLE      : gamma is a near-identity colour curve (like a white-balance shift);
                     P is low-amplitude smooth low-frequency ripple. Both are perceptually
                     under threshold -- a structurally invisible "RGB modification".
  * AUGMENTATION-ROBUST (C3 solved by construction): a global colour map + low-frequency
                     pattern survive RandomCrop / RandomFlip, unlike localised pixel noise.
  * INDEX-INDEPENDENT TEST (C2): gamma's knot values + P weights are SHARED across all
                     images; only the per-image application gamma(x)+P is input-dependent.

Optimisation: proxy CE (flip clean non-target images -> target) under imperceptibility
budgets (P L2-projected; gamma deviation penalised + clamped). Returns (V[3,K] knots,
P[3,H,W] pattern, basis) saved as .npy for the eager-injection apply step.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Low-frequency 2D cosine basis (smooth, invisible, augmentation-robust)
# --------------------------------------------------------------------------- #
def build_lowfreq_basis(size=32, n_basis=48, device='cpu'):
    """Return [n_basis, H, W] of the lowest-frequency separable 2D cosine modes.

    Modes are cos(fx * x) * cos(fy * y) for small (fx, fy), ordered by f = fx+fy.
    Each is L2-normalised. These are smooth (low-frequency) -> visually imperceptible
    when summed at low amplitude, and robust to crop/flip."""
    coords = torch.arange(size, device=device).float() * (math.pi / size)  # [size]
    modes = []
    f = 0
    while len(modes) < n_basis and f <= 2 * size:
        for fx in range(f + 1):
            fy = f - fx
            cx = torch.cos(fx * coords)  # [size]
            cy = torch.cos(fy * coords)  # [size]
            base = cx[:, None] * cy[None, :]  # [H, W] separable
            modes.append(base)
            if len(modes) >= n_basis:
                break
        f += 1
    B = torch.stack(modes, 0)  # [n_basis, H, W]
    B = B / B.flatten(1).norm(dim=1)[:, None, None].clamp(min=1e-8)
    return B  # [n_basis, H, W]


# --------------------------------------------------------------------------- #
# Per-channel piecewise-linear colour LUT (differentiable in knot VALUES V)
# --------------------------------------------------------------------------- #
def apply_lut(x, V):
    """Apply per-channel piecewise-linear map. x: [B,3,H,W] in [0,1]; V: [3,K] knot
    VALUES at knot positions linspace(0,1,K). Returns [B,3,H,W]. Differentiable in V."""
    K = V.shape[1]
    out = torch.empty_like(x)
    for c in range(3):
        idx_f = x[:, c] * (K - 1)            # [B,H,W] in [0, K-1]
        i0 = idx_f.floor().long().clamp(0, K - 2)
        frac = idx_f - i0.float()
        v0 = V[c][i0]                         # gather knot values -> [B,H,W]
        v1 = V[c][i0 + 1]
        out[:, c] = v0 + frac * (v1 - v0)
    return out


def pattern_from_weights(W, B):
    """W: [3, n_basis], B: [n_basis, H, W] -> pattern P: [3, H, W]."""
    return torch.einsum('ck,khw->chw', W, B)


# --------------------------------------------------------------------------- #
# Optimiser
# --------------------------------------------------------------------------- #
def optimize_chroma_trigger(proxy, images, target, device,
                            l2_budget=1.5, steps=8000, lr=0.02,
                            batch_size=128, K=16, n_basis=48,
                            lambda_lut=0.05, lut_max_dev=0.06,
                            exclude_target=True, seed=0, log_every=200,
                            logger=print):
    """Optimise (V, W) so that clamp(apply_lut(x,V) + pattern(W), 0, 1) flips the frozen
    clean proxy to `target` over clean non-target images.

    l2_budget   : hard L2 budget on the pattern P (stealth radius on the additive part).
    lambda_lut  : soft weight on ||V - identity|| (keeps the colour map near-identity).
    lut_max_dev : HARD clamp on per-knot colour deviation |V - pos| -- the invisibility
                  knob (0.06 ~ subtle white-balance shift; the colour map cannot distort
                  beyond this, so ASR must come from the low-freq pattern + subtle colour).
    Returns dict {V:[3,K] numpy, P:[3,H,W] numpy, basis_n, K, info}."""
    from .global_trigger import proxy_argmax  # reuse label helper
    torch.manual_seed(seed)
    images = images.to(device)
    H, Wd = images.shape[-2], images.shape[-1]

    @torch.no_grad()
    def chroma_proxy_asr(eval_n):
        n = min(eval_n, M)
        idx = torch.randperm(M, device=device)[:n]
        correct = 0
        for s in range(0, n, 512):
            sl = idx[s:s + 512]
            x = imgs[sl]
            xa = torch.clamp(apply_lut(x, V) + pattern_from_weights(Wpar, B), 0.0, 1.0)
            correct += proxy(xa).argmax(1).eq(target).sum().item()
        return correct / n

    # restrict to clean non-target images (informative set)
    if exclude_target:
        lbl = proxy_argmax(proxy, images, device)
        pool = torch.where(lbl != target)[0]
        if pool.numel() < batch_size:
            pool = torch.arange(images.shape[0], device=device)
    else:
        pool = torch.arange(images.shape[0], device=device)
    pool = pool.to(device)
    M = pool.numel()
    imgs = images[pool].to(device)              # [M,3,H,W]
    logger('[chroma] optimising: %d clean non-target imgs, L2(pat)=%.3f, steps=%d, '
           'K=%d, n_basis=%d' % (M, l2_budget, steps, K, n_basis))

    B = build_lowfreq_basis(H, n_basis, device)        # [n_basis,H,W]
    pos = torch.linspace(0, 1, K, device=device)       # knot positions
    V = pos.unsqueeze(0).repeat(3, 1).clone().to(device)   # [3,K] identity init
    Wpar = torch.zeros(3, n_basis, device=device)      # pattern weights (0 init)
    V.requires_grad_(True); Wpar.requires_grad_(True)
    opt = torch.optim.Adam([V, Wpar], lr=lr)

    tgt = torch.full((batch_size,), target, dtype=torch.long, device=device)
    g = torch.Generator(device='cpu').manual_seed(seed)
    for step in range(steps):
        idx = torch.randint(0, M, (batch_size,), generator=g)
        x = imgs[idx]
        x_lut = apply_lut(x, V)
        P = pattern_from_weights(Wpar, B)
        x_adv = torch.clamp(x_lut + P, 0.0, 1.0)
        logits = proxy(x_adv)
        # targeted CE: push toward target
        loss_ce = F.cross_entropy(logits, tgt)
        # CW-style margin for universality (optional second term)
        with torch.no_grad():
            pass
        loss_lut = lambda_lut * (V - pos).pow(2).mean()
        loss = loss_ce + loss_lut
        opt.zero_grad(); loss.backward(); opt.step()

        # constraint: project pattern weights so ||P||_2 <= l2_budget (hard stealth radius)
        with torch.no_grad():
            Pn = pattern_from_weights(Wpar, B)
            norm = Pn.flatten().norm()
            if norm > l2_budget:
                Wpar.mul_(l2_budget / norm)
            # HARD invisibility clamp on the colour map: |V - identity| <= lut_max_dev
            V.copy_(pos + (V - pos).clamp(-lut_max_dev, lut_max_dev))

        if (step + 1) % log_every == 0 or step == 0:
            with torch.no_grad():
                pasr = chroma_proxy_asr(min(2000, M))
                Pl2 = pattern_from_weights(Wpar, B).flatten().norm().item()
                lut_dev = (V - pos).abs().max().item()
            logger('[chroma %5d] ce=%.4f |P|=%.3f lutDev=%.3f proxyASR=%.3f'
                   % (step + 1, loss_ce.item(), Pl2, lut_dev, pasr))

    with torch.no_grad():
        Vf = V.detach().cpu().numpy()
        Pf = pattern_from_weights(Wpar, B).detach().cpu().numpy()
        final_asr = chroma_proxy_asr(min(3000, M))
    logger('[chroma] DONE final proxyASR=%.4f |P|=%.3f lutDev=%.3f'
           % (final_asr, np.linalg.norm(Pf), float((V - pos).abs().max().item())))
    return {'V': Vf, 'P': Pf, 'K': K, 'n_basis': n_basis,
            'final_proxy_asr': float(final_asr),
            'info': {'l2_pattern': float(np.linalg.norm(Pf)),
                     'lut_max_dev': float(np.abs(Vf - pos.cpu().numpy()).max())}}

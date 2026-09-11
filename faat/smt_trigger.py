"""SMT: Spectrum-Matched (naturalistic) Trigger.

A free-form additive trigger delta [3,H,W] (necessary for high ASR -- constrained smooth
forms like colour maps / low-freq patterns are empirically too weak; see ChromaTrigger
negative result), BUT optimised so its 2D power spectrum follows the NATURAL image 1/f
envelope. Differentiation from Narcissus:

  * Narcissus delta: adversarial spectrum (whatever maximises proxy CE -- typically
    high-frequency-heavy, carries an anomalous spectral signature detectable by
    frequency-aware defenses, and looks like adversarial noise).
  * SMT delta       : spectrum SHAPED to the dataset's natural 1/f envelope -> looks like
    natural texture/colour variation (perceptually invisible, in-distribution) and has no
    anomalous spectral signature -> harder to detect.

Hypothesis to validate: at matched ASR, SMT is more stealthy (higher SSIM, lower L-inf)
and more defense-evading than Narcissus. The stealth radius is still the L2 budget; the
spectral shaping is the novel constraint.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F


def natural_envelope(size, device='cpu', eps=1.0):
    """1/f natural spectral envelope [H,W] (DC suppressed). Target for |FFT(delta)| shape."""
    fy = torch.fft.fftfreq(size, device=device)
    fx = torch.fft.fftfreq(size, device=device)
    f2 = fx[:, None] ** 2 + fy[None, :] ** 2
    return 1.0 / torch.sqrt(f2 + eps ** 2)  # [H,W], 1/f


def optimize_smt_trigger(proxy, images, target, device,
                         l2_budget=1.5, steps=8000, lr=0.02,
                         batch_size=128, spectral_weight=0.0, spectral_power=2.0,
                         exclude_target=True, seed=0, log_every=200, logger=print):
    """Free-form delta + spectral shaping toward natural 1/f envelope.

    spectral_weight : weight on the spectral-shaping penalty (0 = pure Narcissus).
    spectral_power  : how strongly high frequencies are penalised relative to the 1/f target.
    Returns (delta[3,H,W] numpy, info)."""
    from .global_trigger import proxy_argmax, proxy_asr
    torch.manual_seed(seed)
    images = images.to(device)
    H, W = images.shape[-2], images.shape[-1]

    @torch.no_grad()
    def asr(eval_n):
        return proxy_asr(proxy, images, delta, target, device, eval_n=min(eval_n, M))

    if exclude_target:
        lbl = proxy_argmax(proxy, images, device)
        pool = torch.where(lbl != target)[0]
        if pool.numel() < batch_size:
            pool = torch.arange(images.shape[0], device=device)
    else:
        pool = torch.arange(images.shape[0], device=device)
    pool = pool.to(device)
    M = pool.numel()
    imgs = images[pool]
    logger('[smt] opt: %d non-target imgs, L2=%.3f, steps=%d, spectral_w=%.3f'
           % (M, l2_budget, steps, spectral_weight))

    delta = torch.zeros(3, H, W, device=device, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=lr)
    env = natural_envelope(H, device)            # [H,W] 1/f target (magnitude shape)
    env = env / env.mean()
    tgt = torch.full((batch_size,), target, dtype=torch.long, device=device)
    g = torch.Generator(device='cpu').manual_seed(seed)

    for step in range(steps):
        idx = torch.randint(0, M, (batch_size,), generator=g)
        x = imgs[idx]
        x_adv = torch.clamp(x + delta, 0.0, 1.0)
        loss_ce = F.cross_entropy(proxy(x_adv), tgt)
        # spectral shaping: push |FFT(delta)| magnitude per-channel toward 1/f envelope
        loss_spec = 0.0
        if spectral_weight > 0:
            D = torch.fft.fft2(delta)                  # [3,H,W] complex
            mag = D.abs()                              # [3,H,W]
            # match shape: minimise (mag / env - mean(mag/env))^2  (shape match, not scale)
            ratio = mag / env.unsqueeze(0)
            loss_spec = ((ratio - ratio.mean()) ** 2).mean()
            # plus a soft high-freq penalty to keep it natural
            loss_spec = loss_spec + 0.5 * (mag ** spectral_power).mean()
        loss = loss_ce + spectral_weight * loss_spec
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            n = delta.flatten().norm()
            if n > l2_budget:
                delta.mul_(l2_budget / n)
        if (step + 1) % log_every == 0 or step == 0:
            logger('[smt %5d] ce=%.4f spec=%.4f |d|=%.3f proxyASR=%.3f'
                   % (step + 1, loss_ce.item(),
                      float(loss_spec) if isinstance(loss_spec, torch.Tensor) else 0.0,
                      delta.flatten().norm().item(), asr(2000)))
    df = delta.detach().cpu().numpy()
    logger('[smt] DONE proxyASR=%.4f |d|=%.3f' % (asr(3000), np.linalg.norm(df)))
    return df, {'final_proxy_asr': float(asr(3000)), 'l2': float(np.linalg.norm(df))}

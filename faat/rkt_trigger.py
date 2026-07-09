"""RKT: Resampling-Kernel Trigger.

A *processing-history* trigger: the backdoor is activated by resampling the image with a
secret, proxy-optimized interpolation kernel K at scale s. Unlike additive invisible noise
(Narcissus/ICIT -- which add high-frequency adversarial content and are caught by full-image
NC), RKT is a NON-ADDITIVE global resampling transform:

    x' = Upsample_K( Downsample_s(x) )

The trigger is a fixed shared (s, K) -- index-independent (C2 OK); global & robust to
RandomCrop/Flip (C3 OK). Contribution analog to BppAttack ("optimize trigger via RGB
quantization"): here "optimize trigger via resampling kernel".

Hypotheses (probe-verified, not claims):
  * ASR comparable to BppAttack (structured global signal, learnable).
  * Frequency detector that catches Narcissus (extra high-freq) may MISS RKT (downsample
    REMOVES high-freq -- opposite signature).
  * Full-image NC: may or may not evade (ICIT's input-conditioning did NOT evade NC at 3.24;
    RKT's non-additive transform is a different case -- test honestly, do not pre-claim).
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


def _smooth_init(ksize):
    """Gaussian sum-to-1 initialization for the interpolation kernel."""
    ax = torch.arange(ksize).float() - (ksize - 1) / 2.0
    sigma = ksize / 3.0
    g1d = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
    k = g1d.unsqueeze(1) * g1d.unsqueeze(0)            # outer product -> [k,k] Gaussian
    return k / k.sum()


class RKTTrigger(nn.Module):
    """Resampling trigger: x' = Upsample_K( Downsample_s(x) ). K is a learnable [k,k]
    interpolation kernel (kept ~sum-1 by a regularizer -> brightness-preserving -> stealthy).
    A per-channel `gain` (init 1, reg-kept) lets the optimizer modulate strength per channel
    (mirrors BppAttack's per-channel quantization levels)."""

    def __init__(self, scale=0.7, ksize=5, channels=3):
        super().__init__()
        self.scale = scale
        self.ksize = ksize
        self.channels = channels
        self.kernel = nn.Parameter(_smooth_init(ksize).clone())
        self.gain = nn.Parameter(torch.ones(channels))

    def forward(self, x):
        H = x.shape[-1]
        d = max(1, int(round(H * self.scale)))
        x_down = F.interpolate(x, size=d, mode='area')          # remove high-freq (fixed)
        x_nearest = F.interpolate(x_down, size=H, mode='nearest')
        k = self.kernel.view(1, 1, self.ksize, self.ksize).expand(self.channels, 1, -1, -1).contiguous()
        out = F.conv2d(x_nearest, k, padding=self.ksize // 2, groups=self.channels)
        out = out * self.gain.view(1, -1, 1, 1)
        return torch.clamp(out, 0.0, 1.0)


def optimize_rkt_trigger(proxy, images, target, device, scale=0.7, ksize=5,
                         steps=3000, lr=5e-3, batch_size=128, stealth_lam=1e-2,
                         exclude_target=True, seed=0, log_every=300, logger=print):
    """Optimize the resampling kernel K so proxy(RKT(x)) -> target over clean non-target
    images. Saves K+gain+meta to `save_trigger`. Returns info dict."""
    from .global_trigger import proxy_argmax
    torch.manual_seed(seed)
    images = images.to(device)
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
    logger('[rkt] optimizing resampling kernel: %d non-target imgs, scale=%.2f, ksize=%d, steps=%d'
           % (M, scale, ksize, steps))

    trig = RKTTrigger(scale=scale, ksize=ksize).to(device)
    init_k = _smooth_init(ksize).to(device)
    opt = torch.optim.Adam([trig.kernel, trig.gain], lr=lr)
    tgt = torch.full((batch_size,), target, dtype=torch.long, device=device)
    g = torch.Generator(device='cpu').manual_seed(seed)

    @torch.no_grad()
    def proxy_asr(eval_n):
        n = min(eval_n, M)
        idx = torch.randperm(M, device=device)[:n]
        correct = 0
        for s in range(0, n, 512):
            x = imgs[idx[s:s + 512]]
            correct += proxy(trig(x)).argmax(1).eq(target).sum().item()
        return correct / n

    for step in range(steps):
        idx = torch.randint(0, M, (batch_size,), generator=g)
        x = imgs[idx]
        xp = trig(x)
        k = trig.kernel
        reg = (stealth_lam * (k - init_k).pow(2).sum()
               + stealth_lam * (k.sum() - 1.0).pow(2)
               + stealth_lam * (trig.gain - 1.0).pow(2).sum())
        loss = F.cross_entropy(proxy(xp), tgt) + reg
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % log_every == 0 or step == 0:
            logger('[rkt %5d] loss=%.4f proxyASR=%.3f' % (step + 1, loss.item(), proxy_asr(2000)))
    final = proxy_asr(3000)
    logger('[rkt] DONE proxyASR=%.4f' % final)
    return {'trig': trig, 'final_proxy_asr': float(final), 'scale': scale, 'ksize': ksize}


def save_rkt_trigger(trig, save_trigger, scale, ksize, final_asr):
    os.makedirs(save_trigger, exist_ok=True)
    torch.save({'kernel': trig.kernel.detach().cpu(),
                'gain': trig.gain.detach().cpu(),
                'scale': scale, 'ksize': ksize},
               os.path.join(save_trigger, 'rkt.pth'))
    with open(os.path.join(save_trigger, 'meta.txt'), 'w') as f:
        f.write('RKT resampling-kernel trigger. proxyASR=%.4f scale=%.2f ksize=%d\n'
                % (final_asr, scale, ksize))


def load_rkt_trigger(save_trigger, device):
    ck = torch.load(os.path.join(save_trigger, 'rkt.pth'), map_location=device)
    trig = RKTTrigger(scale=ck['scale'], ksize=ck['ksize']).to(device)
    trig.kernel.data = ck['kernel'].to(device)
    trig.gain.data = ck['gain'].to(device)
    trig.eval()
    for p in trig.parameters():
        p.requires_grad_(False)
    return trig

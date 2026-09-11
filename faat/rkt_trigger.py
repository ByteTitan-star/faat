"""RKT: Resampling-Kernel Trigger.

A *processing-history* trigger: the backdoor is activated by resampling the image with a
secret, proxy-optimized interpolation kernel K at scale s. Unlike additive invisible noise
(Narcissus/ICIT -- which add high-frequency adversarial content and are caught by full-image
NC), RKT is a NON-ADDITIVE global resampling transform:

    x' = Upsample_K( Downsample_s(x) )

K is HARD-NORMALIZED to sum=1 inside forward -> always a valid, BRIGHTNESS-PRESERVING
interpolation kernel (cannot amplify -> cannot cheat via a per-channel gain, which the first
implementation did: it zeroed the green channel and crashed SSIM to 0.34). There is NO gain
parameter: a resampling must preserve brightness by definition; the signal comes purely from
kernel SHAPE + downsample scale.

The trigger is a fixed shared (s, K) -- index-independent (C2 OK); global & robust to
RandomCrop/Flip (C3 OK). Contribution analog to BppAttack ("optimize trigger via RGB
quantization"): here "optimize trigger via resampling kernel".

Open question (this file's reason for existing): can a brightness-preserving resampling
reach high proxyASR at stealthy SSIM>=0.9? If not, RKT hits the same wall as ChromaTrigger
(low-amplitude smooth transforms don't flip classifiers). The sweep answers this.
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
    """Resampling trigger: x' = Upsample_K( Downsample_s(x) ). K is a learnable [k,k] kernel,
    HARD-normalized to sum=1 in forward (valid interpolator, brightness-preserving, no gain)."""

    def __init__(self, scale=0.7, ksize=5, channels=3):
        super().__init__()
        self.scale = scale
        self.ksize = ksize
        self.channels = channels
        self.kernel = nn.Parameter(_smooth_init(ksize).clone())

    def _norm_kernel(self):
        k = self.kernel
        k = k / (k.sum() + 1e-8)                            # HARD sum=1
        return k.view(1, 1, self.ksize, self.ksize).expand(self.channels, 1, -1, -1).contiguous()

    def forward(self, x):
        H = x.shape[-1]
        d = max(1, int(round(H * self.scale)))
        x_down = F.interpolate(x, size=d, mode='area')     # remove high-freq (fixed)
        x_nearest = F.interpolate(x_down, size=H, mode='nearest')
        out = F.conv2d(x_nearest, self._norm_kernel(), padding=self.ksize // 2, groups=self.channels)
        return torch.clamp(out, 0.0, 1.0)


def optimize_rkt_trigger(proxy, images, target, device, scale=0.7, ksize=5,
                         steps=3000, lr=5e-3, batch_size=128, stealth_lam=1e-2,
                         exclude_target=True, seed=0, log_every=300, logger=print):
    """Optimize the resampling kernel K so proxy(RKT(x)) -> target over clean non-target
    images. Returns info dict with the trained trigger."""
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
    init_k = _smooth_init(ksize).to(device)                # sum=1 reference for stealth reg
    opt = torch.optim.Adam([trig.kernel], lr=lr)
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
        k_eff = trig.kernel / (trig.kernel.sum() + 1e-8)
        reg = stealth_lam * (k_eff - init_k).pow(2).sum()           # keep kernel ~smooth
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
                'scale': scale, 'ksize': ksize},
               os.path.join(save_trigger, 'rkt.pth'))
    with open(os.path.join(save_trigger, 'meta.txt'), 'w') as f:
        f.write('RKT resampling-kernel trigger (no-gain, sum=1). proxyASR=%.4f scale=%.2f ksize=%d\n'
                % (final_asr, scale, ksize))


def load_rkt_trigger(save_trigger, device):
    ck = torch.load(os.path.join(save_trigger, 'rkt.pth'), map_location=device)
    trig = RKTTrigger(scale=ck['scale'], ksize=ck['ksize']).to(device)
    trig.kernel.data = ck['kernel'].to(device)
    trig.eval()
    for p in trig.parameters():
        p.requires_grad_(False)
    return trig

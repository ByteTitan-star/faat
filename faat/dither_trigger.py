"""Dither-Trigger: ordered-dither quantization as a structure-PRESERVING backdoor trigger.

Why this exists: RKT (resampling) FAILED on CIFAR (faat/rkt_trigger.py) because downsampling
DESTROYS spatial structure -> visible on 32x32 (SSIM crashed to 0.5). Dithering is
quantization (structure-PRESERVING, like BppAttack) but with a learnable tileable spatial
threshold matrix D: the quantization threshold varies per pixel position. This (a) keeps
stealth (quantization -> bounded per-pixel perturbation ~1 step), (b) adds a learnable
spatial signature richer than BppAttack's UNIFORM per-channel quantization. Hypothesis:
dither can reach high ASR at SSIM>=0.9 where RKT could not.

  x' = clamp( round( (x + sigmoid(D_tiled)*step) / step ) * step, 0, 1 ),  step = 1/levels
  D is per-channel [tile x tile]; the sigmoid bounds the dither offset to [0, step) (<=1 bin).
  STE round (z_q = z + (round(z)-z).detach) gives gradient to D. At uniform D this is plain
  quantization (BppAttack-like); the learned spatial D IS the trigger signal.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class DitherTrigger(nn.Module):
    def __init__(self, tile=4, levels=16, channels=3):
        super().__init__()
        self.tile = tile
        self.levels = levels
        self.channels = channels
        # small random init -> sigmoid ~0.5 +/-, i.e. a near-uniform half-step dither to start
        self.dither = nn.Parameter(torch.randn(channels, tile, tile) * 0.1)

    def _tiled(self, H, W):
        rep_h = (H + self.tile - 1) // self.tile
        rep_w = (W + self.tile - 1) // self.tile
        return self.dither.repeat(1, rep_h, rep_w)[:, :H, :W]      # [c,H,W]

    def forward(self, x):
        B, c, H, W = x.shape
        step = 1.0 / self.levels
        d_full = self._tiled(H, W).unsqueeze(0).expand(B, c, H, W)
        offset = torch.sigmoid(d_full) * step                       # bounded in [0, step)
        z = (x + offset) / step
        z_q = z + (torch.round(z) - z).detach()                     # STE round
        return torch.clamp(z_q * step, 0.0, 1.0)


def optimize_dither_trigger(proxy, images, target, device, tile=4, levels=16,
                            steps=2000, lr=5e-3, batch_size=128,
                            exclude_target=True, seed=0, log_every=400, logger=print):
    """Optimize the dither matrix D so proxy(dither(x)) -> target over clean non-target images."""
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
    logger('[dither] optimizing: %d non-target imgs, tile=%d, levels=%d, steps=%d'
           % (M, tile, levels, steps))

    trig = DitherTrigger(tile=tile, levels=levels).to(device)
    opt = torch.optim.Adam([trig.dither], lr=lr)
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
        loss = F.cross_entropy(proxy(trig(x)), tgt)
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % log_every == 0 or step == 0:
            logger('[dither %5d] ce=%.4f proxyASR=%.3f' % (step + 1, loss.item(), proxy_asr(2000)))
    final = proxy_asr(3000)
    logger('[dither] DONE proxyASR=%.4f' % final)
    return {'trig': trig, 'final_proxy_asr': float(final), 'tile': tile, 'levels': levels}


def save_dither_trigger(trig, save_trigger, tile, levels, final_asr):
    os.makedirs(save_trigger, exist_ok=True)
    torch.save({'dither': trig.dither.detach().cpu(), 'tile': tile, 'levels': levels},
               os.path.join(save_trigger, 'dither.pth'))
    with open(os.path.join(save_trigger, 'meta.txt'), 'w') as f:
        f.write('Dither trigger. proxyASR=%.4f tile=%d levels=%d\n' % (final_asr, tile, levels))


def load_dither_trigger(save_trigger, device):
    ck = torch.load(os.path.join(save_trigger, 'dither.pth'), map_location=device)
    trig = DitherTrigger(tile=ck['tile'], levels=ck['levels']).to(device)
    trig.dither.data = ck['dither'].to(device)
    trig.eval()
    for p in trig.parameters():
        p.requires_grad_(False)
    return trig

"""ICIT: Input-Conditioned Imperceptible Trigger.

A small generator g(x) produces a per-image BOUNDED perturbation, trained (frozen clean
proxy) so that proxy(clamp(x + g(x), 0, 1)) -> target. g is SHARED across all images
(satisfies C2: index-independent test trigger), but the effective trigger g(x) VARIES per
image -- so Neural Cleanse's universal-delta reverse-engineering cannot capture it with
one small delta per class -> evades NC (which catches Narcissus, a universal trigger).

This is the novel contribution chain:
  * Narcissus (imperceptible UNIVERSAL) evades AC/SS/STRIP/FP but is caught by NC.
  * ICIT (imperceptible INPUT-CONDITIONED) matches Narcissus ASR/stealth AND evades NC.

The generator is trained once (offline), saved as a state_dict, and applied (frozen) at
eager-injection time to every poison/test image.
"""
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ICGenerator(nn.Module):
    """Image -> bounded imperceptible perturbation [3,H,W], per-image L2-projected to <=budget."""

    def __init__(self, channels=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(),
            nn.Conv2d(channels, 3, 3, padding=1))

    def forward(self, x, budget):
        d = self.net(x)
        flat = d.flatten(1)
        n = flat.norm(dim=1, keepdim=True).clamp(min=1e-8)
        scale = torch.clamp(budget / n, max=1.0)   # per-image L2 projection to <= budget
        return d * scale.view(-1, 1, 1, 1)


def optimize_ic_trigger(proxy, images, target, device,
                        budget=1.5, steps=3000, lr=5e-4, batch_size=128,
                        exclude_target=True, seed=0, log_every=300, logger=print):
    """Train ICGenerator g so proxy(clamp(x+g(x,budget),0,1)) -> target over clean non-target
    images. Saves g.state_dict() + meta to `save_trigger`. Returns info dict."""
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
    logger('[icit] training input-conditioned generator: %d non-target imgs, budget=%.3f, steps=%d'
           % (M, budget, steps))

    gen = ICGenerator().to(device)
    opt = torch.optim.Adam(gen.parameters(), lr=lr)
    tgt = torch.full((batch_size,), target, dtype=torch.long, device=device)
    g = torch.Generator(device='cpu').manual_seed(seed)

    @torch.no_grad()
    def proxy_asr(eval_n):
        n = min(eval_n, M)
        idx = torch.randperm(M, device=device)[:n]
        correct = 0
        for s in range(0, n, 512):
            x = imgs[idx[s:s + 512]]
            correct += proxy(torch.clamp(x + gen(x, budget), 0, 1)).argmax(1).eq(target).sum().item()
        return correct / n

    for step in range(steps):
        idx = torch.randint(0, M, (batch_size,), generator=g)
        x = imgs[idx]
        xp = torch.clamp(x + gen(x, budget), 0, 1)
        loss = F.cross_entropy(proxy(xp), tgt)
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % log_every == 0 or step == 0:
            logger('[icit %5d] ce=%.4f proxyASR=%.3f' % (step + 1, loss.item(), proxy_asr(2000)))
    final = proxy_asr(3000)
    logger('[icit] DONE proxyASR=%.4f' % final)
    return {'gen': gen, 'final_proxy_asr': float(final), 'budget': budget}


def save_ic_trigger(gen, save_trigger, budget, final_asr):
    os.makedirs(save_trigger, exist_ok=True)
    torch.save({'state_dict': gen.state_dict(), 'budget': budget}, os.path.join(save_trigger, 'gen.pth'))
    with open(os.path.join(save_trigger, 'meta.txt'), 'w') as f:
        f.write('ICIT input-conditioned trigger. proxyASR=%.4f budget=%.3f\n' % (final_asr, budget))


def load_ic_generator(save_trigger, device):
    ck = torch.load(os.path.join(save_trigger, 'gen.pth'), map_location=device)
    gen = ICGenerator().to(device)
    gen.load_state_dict(ck['state_dict'])
    gen.eval()
    for p in gen.parameters():
        p.requires_grad_(False)
    return gen, float(ck['budget'])

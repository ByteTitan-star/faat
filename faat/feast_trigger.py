"""FEAST triggers: two trigger parameterizations paired with Feature Starvation (feast_starve.py).

User's new angle (2026-07-10): instead of amplifying the trigger to compete with the target image's
natural features (Narcissus wall), first STARVE the natural target features (feast_starve.py), then
inject a weak trigger the model is forced to learn. Two trigger modes:
  * 'phase' (user's design): universal low-freq Fourier-PHASE shift. Hypothesis: transfers CNN->ViT,
    imperceptible. (Smoke test: walls -- phase carries structure, so it is highly visible AND a weak
    adversarial direction. Kept for an honest controlled test.)
  * 'pixel' (added to ISOLATE the starvation mechanism): universal pixel delta (Narcissus-style),
    a known adversarial-effective direction. Pairing starvation with an effective trigger isolates
    whether feature-starvation itself breaks the stealth/ASR wall.

Both modes produce one shared trigger tensor + a 'mode' field, applied identically at train (after
starvation) and at test (no starvation, C2: index-independent).
"""
import os
import torch
import torch.nn.functional as F


# ----------------------------- phase mode -----------------------------

def _neg_index(n):
    return (-torch.arange(n)) % n


def make_low_freq_mask(h, w, radius, device):
    uu = torch.arange(h, device=device).float()
    vv = torch.arange(w, device=device).float()
    uu = torch.minimum(uu, h - uu)
    vv = torch.minimum(vv, w - vv)
    r2 = uu[:, None] ** 2 + vv[None, :] ** 2
    return (r2 < float(radius) ** 2).float()


def dphi_from_theta(theta, mask):
    c, h, w = theta.shape
    neg_u = _neg_index(h).to(theta.device)
    neg_v = _neg_index(w).to(theta.device)
    neg = theta[:, neg_u][:, :, neg_v]
    dphi = 0.5 * (theta - neg)              # antisymmetric -> ifft stays real
    dphi = dphi * mask.unsqueeze(0)
    dphi[:, 0, 0] = 0.0
    return dphi


def phase_apply(x, dphi):
    F0 = torch.fft.fft2(x, norm='ortho')
    Fp = F0 * torch.exp(1j * dphi.unsqueeze(0).to(x.dtype))
    return torch.clamp(torch.fft.ifft2(Fp, norm='ortho').real, 0.0, 1.0)


# ----------------------------- pixel mode -----------------------------

def pixel_apply(x, delta):
    return torch.clamp(x + delta.unsqueeze(0).to(x.dtype), 0.0, 1.0)


# ----------------------------- unified apply / artifact -----------------------------

def feast_apply(x, mode, trig):
    trig = trig.to(x.device)
    return phase_apply(x, trig) if mode == 'phase' else pixel_apply(x, trig)


def optimize_feast_phase(proxy, images, target, device,
                         phi_max=0.6, radius=8, steps=3000, lr=5e-2,
                         batch_size=128, exclude_target=True, seed=0,
                         log_every=300, logger=print):
    from .global_trigger import proxy_argmax
    torch.manual_seed(seed)
    images = images.to(device)
    _, c, h, w = images.shape
    mask = make_low_freq_mask(h, w, radius, device)
    pool = _non_target_pool(proxy, images, target, batch_size, exclude_target, device)
    M = pool.numel(); imgs = images[pool]
    logger('[feast-phase] optimizing: %d non-target imgs, phi_max=%.2f rad, radius=%d, steps=%d'
           % (M, phi_max, radius, steps))
    theta = torch.zeros(c, h, w, device=device, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=lr)
    tgt = torch.full((batch_size,), target, dtype=torch.long, device=device)
    g = torch.Generator(device='cpu').manual_seed(seed)

    @torch.no_grad()
    def asr(eval_n):
        n = min(eval_n, M); idx = torch.randperm(M, device=device)[:n]; correct = 0
        dphi = dphi_from_theta(theta, mask)
        for s in range(0, n, 512):
            correct += proxy(phase_apply(imgs[idx[s:s + 512]], dphi)).argmax(1).eq(target).sum().item()
        return correct / n

    for step in range(steps):
        idx = torch.randint(0, M, (batch_size,), generator=g)
        dphi = dphi_from_theta(theta, mask)
        loss = F.cross_entropy(proxy(phase_apply(imgs[idx], dphi)), tgt)
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            theta.clamp_(-phi_max, phi_max)
        if (step + 1) % log_every == 0 or step == 0:
            logger('[feast-phase %5d] ce=%.4f proxyASR=%.3f' % (step + 1, loss.item(), asr(2000)))
    final = asr(3000)
    logger('[feast-phase] DONE proxyASR=%.4f |dPhi|_max=%.3f' % (final, dphi_from_theta(theta, mask).abs().max().item()))
    return {'mode': 'phase', 'trig': dphi_from_theta(theta, mask).detach(),
            'final_proxy_asr': float(final), 'phi_max': phi_max, 'radius': radius}


def optimize_feast_pixel(proxy, images, target, device,
                         budget=1.0, steps=3000, lr=5e-3,
                         batch_size=128, exclude_target=True, seed=0,
                         log_every=300, logger=print):
    """Universal pixel delta (Narcissus-style), L2-projected to <= budget. An effective adversarial
    direction -- paired with starvation to ISOLATE the starvation mechanism from trigger-form."""
    from .global_trigger import proxy_argmax
    torch.manual_seed(seed)
    images = images.to(device)
    _, c, h, w = images.shape
    pool = _non_target_pool(proxy, images, target, batch_size, exclude_target, device)
    M = pool.numel(); imgs = images[pool]
    logger('[feast-pixel] optimizing: %d non-target imgs, L2 budget=%.2f, steps=%d' % (M, budget, steps))
    delta = torch.zeros(c, h, w, device=device, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=lr)
    tgt = torch.full((batch_size,), target, dtype=torch.long, device=device)
    g = torch.Generator(device='cpu').manual_seed(seed)

    @torch.no_grad()
    def asr(eval_n):
        n = min(eval_n, M); idx = torch.randperm(M, device=device)[:n]; correct = 0
        for s in range(0, n, 512):
            correct += proxy(pixel_apply(imgs[idx[s:s + 512]], _l2_proj(delta, budget))).argmax(1).eq(target).sum().item()
        return correct / n

    for step in range(steps):
        idx = torch.randint(0, M, (batch_size,), generator=g)
        dp = _l2_proj(delta, budget)
        loss = F.cross_entropy(proxy(pixel_apply(imgs[idx], dp)), tgt)
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % log_every == 0 or step == 0:
            logger('[feast-pixel %5d] ce=%.4f proxyASR=%.3f' % (step + 1, loss.item(), asr(2000)))
    final = asr(3000)
    logger('[feast-pixel] DONE proxyASR=%.4f |delta|2=%.3f' % (final, _l2_proj(delta, budget).norm().item()))
    return {'mode': 'pixel', 'trig': _l2_proj(delta, budget).detach(),
            'final_proxy_asr': float(final), 'budget': budget}


def _l2_proj(delta, budget):
    n = delta.flatten().norm()
    scale = (budget / n.clamp(min=1e-8)).clamp(max=1.0)
    return delta * scale


def _non_target_pool(proxy, images, target, batch_size, exclude_target, device):
    from .global_trigger import proxy_argmax
    if exclude_target:
        lbl = proxy_argmax(proxy, images, device)
        pool = torch.where(lbl != target)[0]
        if pool.numel() < batch_size:
            pool = torch.arange(images.shape[0], device=device)
    else:
        pool = torch.arange(images.shape[0], device=device)
    return pool.to(device)


def save_feast_trigger(info, save_trigger):
    os.makedirs(save_trigger, exist_ok=True)
    rec = {'mode': info['mode'], 'trig': info['trig'].cpu()}
    for k in ('phi_max', 'radius', 'budget'):
        if k in info:
            rec[k] = info[k]
    torch.save(rec, os.path.join(save_trigger, 'feast.pth'))
    with open(os.path.join(save_trigger, 'meta.txt'), 'w') as f:
        f.write('FEAST trigger mode=%s proxyASR=%.4f %s\n'
                % (info['mode'], info['final_proxy_asr'],
                   ' '.join('%s=%s' % (k, info[k]) for k in ('phi_max', 'radius', 'budget') if k in info)))


def load_feast_artifact(save_trigger, device):
    ck = torch.load(os.path.join(save_trigger, 'feast.pth'), map_location=device)
    return ck['mode'], ck['trig'].to(device)

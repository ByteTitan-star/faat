"""PAT: Perceptually-Allocated Trigger (my own innovation, not a reproduction).

INSIGHT: trigger budget should be ALLOCATED by perceptual saliency (JND), not uniformly.
Narcissus uses a uniform L2 noise budget; BppAttack uses uniform per-channel quantization.
PAT clips a universal perturbation to alpha * JND(x) per pixel -- so the trigger is below
the just-noticeable-difference everywhere (invisible by construction), while concentrating
budget in textured regions where (a) humans can't see it (contrast masking) AND (b) the
model has rich features (effective). Same total perceived stealth -> higher ASR than uniform.

Mechanism: delta is universal [3,H,W]; effective per-image trigger = clamp(delta, -alpha*JND(x),
+alpha*JND(x)). JND(x) = jmin + (jmax-jmin)*normalize(local_contrast(x)) (Sobel). Textured
regions -> high JND -> more budget allowed. Optimize delta (proxy CE) under this JND-clip.

C2: trigger op = clamp(delta, alpha*JND(x)) is a shared index-independent function of x.
C3: delta global (frequency-spread) survives RandomCrop/Flip.
"""
import os
import torch
import torch.nn.functional as F


def jnd_map(x, jmin=0.02, jmax=0.25):
    """Per-image JND map [B,1,H,W] in [jmin,jmax]. Higher in textured (high-contrast) regions."""
    gray = x.mean(1, keepdim=True)
    sobel = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=x.device).view(1, 1, 3, 3)
    gx = F.conv2d(gray, sobel, padding=1)
    gy = F.conv2d(gray, sobel.transpose(2, 3), padding=1)
    contrast = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)
    cmin = contrast.amin(dim=(2, 3), keepdim=True)
    cmax = contrast.amax(dim=(2, 3), keepdim=True)
    cn = (contrast - cmin) / (cmax - cmin + 1e-8)
    return jmin + (jmax - jmin) * cn


def pat_apply(x, delta, alpha, jnd=None):
    if jnd is None:
        jnd = jnd_map(x)
    bound = alpha * jnd                                  # [B,1,H,W]
    delta_eff = torch.clamp(delta, -bound, bound)       # clip universal delta to per-image JND
    return torch.clamp(x + delta_eff, 0.0, 1.0)


def optimize_pat_trigger(proxy, images, target, device, alpha=1.0, steps=3000, lr=5e-3,
                         batch_size=128, exclude_target=True, seed=0, log_every=300, logger=print):
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
    pool = pool.to(device); M = pool.numel(); imgs = images[pool]
    logger('[pat] optimizing: %d non-target imgs, alpha=%.2f, steps=%d' % (M, alpha, steps))

    delta = torch.zeros(3, 32, 32, device=device, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=lr)
    tgt = torch.full((batch_size,), target, dtype=torch.long, device=device)
    g = torch.Generator(device='cpu').manual_seed(seed)

    @torch.no_grad()
    def proxy_asr(eval_n):
        n = min(eval_n, M); idx = torch.randperm(M, device=device)[:n]; correct = 0
        for s in range(0, n, 512):
            x = imgs[idx[s:s + 512]]
            correct += proxy(pat_apply(x, delta, alpha)).argmax(1).eq(target).sum().item()
        return correct / n

    for step in range(steps):
        idx = torch.randint(0, M, (batch_size,), generator=g)
        x = imgs[idx]
        loss = F.cross_entropy(proxy(pat_apply(x, delta, alpha)), tgt)
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % log_every == 0 or step == 0:
            logger('[pat %5d] ce=%.4f proxyASR=%.3f' % (step + 1, loss.item(), proxy_asr(2000)))
    final = proxy_asr(3000)
    logger('[pat] DONE proxyASR=%.4f' % final)
    return {'delta': delta.detach(), 'final_proxy_asr': float(final), 'alpha': alpha}


def save_pat_trigger(delta, save_trigger, alpha, final_asr):
    os.makedirs(save_trigger, exist_ok=True)
    torch.save({'delta': delta.cpu(), 'alpha': alpha}, os.path.join(save_trigger, 'pat.pth'))
    with open(os.path.join(save_trigger, 'meta.txt'), 'w') as f:
        f.write('PAT perceptually-allocated trigger. proxyASR=%.4f alpha=%.2f\n' % (final_asr, alpha))


def load_pat_delta(save_trigger, device):
    ck = torch.load(os.path.join(save_trigger, 'pat.pth'), map_location=device)
    return ck['delta'].to(device), float(ck['alpha'])

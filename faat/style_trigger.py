"""Style-based trigger (user direction 2026-07-10, the one non-obviously-prior of the three).

INSIGHT: instead of a pixel-adversarial direction (Narcissus/PAT/phase -- all walled on the
adversarial-distance floor), use a TEXTURE/STYLE statistic as the trigger signal. Optimize a
universal delta so the Gram matrix of the proxy's deep feature map of (x+delta) matches the TARGET
class's mean Gram matrix -> the triggered sample carries the target class's texture/statistical
signature -> the victim learns "this texture style -> target".

This is a DIFFERENT basis than pixel-CE: it is 2nd-order feature-statistic alignment (Gatys-style
style loss). Predicted outcomes (stated up front for honesty):
  * if it works via target-class feature-distribution alignment, it is the SGBA family (prior) --
    PA-ICT already showed target-class feature alignment works but = SGBA;
  * if it walls, it confirms (again) that perceptual/global perturbation bases are orthogonal to
    adversarial effectiveness (the Meta wall).

Two modes mirror PA-ICT:
  * 'gram_ce' : minimize Gram-style loss + proxy CE toward target (main).
  * 'gram'    : pure Gram alignment, no CE (alignment-alone ablation, like pure-PA-ICT).

C2: style_apply(x, delta) is a shared index-independent function (universal delta).
C3: delta is global/low-freq-ish (texture stats are spatially distributed) -> survives RandomCrop.
"""
import os
import torch
import torch.nn.functional as F


def layer4_feat(proxy, x):
    """Replicate proxy forward up to layer4 -> [B,512,h,w] feature map (pre-pool)."""
    out = F.relu(proxy.bn1(proxy.conv1(x)))
    out = proxy.layer1(out)
    out = proxy.layer2(out)
    out = proxy.layer3(out)
    out = proxy.layer4(out)
    return out


def gram(feat):
    """Gram matrix of a [B,C,H,W] feature map -> [B,C,C] (normalized)."""
    b, c, h, w = feat.shape
    f = feat.view(b, c, h * w)
    g = torch.bmm(f, f.transpose(1, 2))
    return g / (c * h * w)


@torch.no_grad()
def compute_target_gram(proxy, target_imgs, device, batch=256):
    """Mean Gram matrix over clean target-class images -> [C,C]."""
    gsum = None; n = 0
    for s in range(0, target_imgs.shape[0], batch):
        x = target_imgs[s:s + batch].to(device)
        gg = gram(layer4_feat(proxy, x))
        gsum = gg.sum(0) if gsum is None else gsum + gg.sum(0)
        n += gg.shape[0]
    g = (gsum / n).detach()
    # per-matrix standardization scale (so the loss is O(1), not ~1e-7 from tiny raw Gram values)
    mean = g.mean(); std = g.std() + 1e-8
    return ((g - mean) / std).detach()  # standardized target gram


def gram_norm(feat):
    """Standardized Gram matrix (zero-mean, unit-std per matrix) to match standardized target."""
    g = gram(feat)
    return (g - g.mean(dim=(1, 2), keepdim=True)) / (g.std(dim=(1, 2), keepdim=True) + 1e-8)


def style_apply(x, delta):
    return torch.clamp(x + delta.unsqueeze(0).to(x.dtype), 0.0, 1.0)


def optimize_style_trigger(proxy, images, labels, target, device,
                           mode='gram_ce', budget=1.0, steps=3000, lr=5e-3,
                           batch_size=128, seed=0, log_every=300, logger=print):
    """Optimize universal delta [3,32,32] (L2<=budget). gram_ce = Gram-align + CE; gram = Gram only.
    images/labels: full clean train set (used to form non-target pool + target-class Gram target)."""
    torch.manual_seed(seed)
    images = images.to(device); labels = labels.to(device)
    tgt_imgs = images[labels == target]
    target_gram = compute_target_gram(proxy, tgt_imgs, device)
    # also a random-class gram for a control column (logged, not used in loss)
    other = torch.where(labels != target)[0]
    pool = other if other.numel() >= batch_size else torch.arange(images.shape[0], device=device)
    M = pool.numel(); imgs = images[pool]
    logger('[style] mode=%s budget=%.2f steps=%d | target_gram from %d imgs, pool %d non-target'
           % (mode, budget, steps, tgt_imgs.shape[0], M))

    _, c, h, w = images.shape
    delta = torch.zeros(c, h, w, device=device, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=lr)
    tgt = torch.full((batch_size,), target, dtype=torch.long, device=device)
    g = torch.Generator(device='cpu').manual_seed(seed)

    def proj(d):
        n = d.flatten().norm()
        return d * (budget / n.clamp(min=1e-8)).clamp(max=1.0)

    @torch.no_grad()
    def asr(eval_n):
        n = min(eval_n, M); idx = torch.randperm(M, device=device)[:n]; correct = 0
        for s in range(0, n, 512):
            correct += proxy(style_apply(imgs[idx[s:s + 512]], proj(delta))).argmax(1).eq(target).sum().item()
        return correct / n

    for step in range(steps):
        idx = torch.randint(0, M, (batch_size,), generator=g)
        x = imgs[idx]
        dp = proj(delta)
        xp = style_apply(x, dp)
        feat = layer4_feat(proxy, xp)
        l_gram = (gram_norm(feat) - target_gram.unsqueeze(0)).pow(2).mean()
        if mode == 'gram_ce':
            l_ce = F.cross_entropy(proxy(xp), tgt)
            loss = l_gram + 0.05 * l_ce
        else:
            loss = l_gram
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % log_every == 0 or step == 0:
            logger('[style %5d] %s gram=%.5f proxyASR=%.3f' %
                   (step + 1, mode, l_gram.item(), asr(2000)))
    final = asr(3000)
    logger('[style] DONE mode=%s proxyASR=%.4f |delta|2=%.3f' % (mode, final, proj(delta).norm().item()))
    return {'mode': mode, 'trig': proj(delta).detach(), 'final_proxy_asr': float(final), 'budget': budget}


def save_style_trigger(info, save_trigger):
    os.makedirs(save_trigger, exist_ok=True)
    torch.save({'mode': info['mode'], 'trig': info['trig'].cpu(), 'budget': info['budget']},
               os.path.join(save_trigger, 'style.pth'))
    with open(os.path.join(save_trigger, 'meta.txt'), 'w') as f:
        f.write('Style trigger mode=%s proxyASR=%.4f budget=%.2f\n'
                % (info['mode'], info['final_proxy_asr'], info['budget']))


def load_style_trigger(save_trigger, device):
    ck = torch.load(os.path.join(save_trigger, 'style.pth'), map_location=device)
    return ck['mode'], ck['trig'].to(device)

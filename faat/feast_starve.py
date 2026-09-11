"""FEAST Feature Starvation: pre-destroy the natural target-class features of each poison image.

User's core new idea: in clean-label attacks the poison image ALREADY carries strong natural
target-class features (robust features), and the model's simplicity bias learns those and ignores
a weak trigger -- this is the "wall". Instead of amplifying the trigger (hits the perceptual
floor), FEAST removes the natural signal: push each poison image AWAY from y_target in the
frozen proxy's feature/logit space via an untargeted adversarial step, under a SMALL perceptual
budget. The image still carries the y_target label, but the model can no longer extract natural
target features from it -> "feature starvation" -> the model is forced to learn the (weak) phase
trigger as the only available shortcut to fit the label.

  x_starved = argmax_{x'} CE(f(x'), y_target)  s.t. ||x'-x||_inf <= eps   (L_inf proxy for LPIPS)

Starvation is applied ONLY to train poison images at eager-injection time (C1). It is NOT applied
at test time (C2) -- the backdoor is the phase trigger; starvation is a training-time teaching aid.
"""
import torch
import torch.nn.functional as F


@torch.no_grad()
def _proxy_pred_target_counts(proxy, x, target):
    out = proxy(x)
    return out.argmax(1).eq(target).sum().item(), out.shape[0]


def starve_batch(x, proxy, target, eps, steps=30, step_size=None, device='cuda', logger=None):
    """Untargeted adversarial push: maximize CE(proxy(x+delta), target) so features move AWAY from
    the target class, under an L_inf ball of radius eps. x: [B,C,H,W] in [0,1]. Returns x_starved.
    eps=0 -> returns x unchanged (ablation: no starvation)."""
    x = x.to(device)
    if eps <= 0:
        return x
    if step_size is None:
        step_size = eps / 8.0 * 2.5        # ~2.5x the per-step L_inf of a eps/8 schedule
    tgt = torch.full((x.shape[0],), target, dtype=torch.long, device=device)
    with torch.enable_grad():              # robust to any global grad-state leak from callers
        delta = torch.zeros_like(x, requires_grad=True)
        for _ in range(steps):
            xp = torch.clamp(x + delta, 0.0, 1.0)
            loss = F.cross_entropy(proxy(xp), tgt)      # ASCEND -> push away from target
            grad, = torch.autograd.grad(loss, delta)
            with torch.no_grad():
                delta.add_(step_size * grad.sign()).clamp_(-eps, eps)  # in-place: keep leaf + grad
        return torch.clamp(x + delta, 0.0, 1.0).detach()


def starve_dataset_images(dataset, indices, proxy, target, eps, steps=30,
                          step_size=None, device='cuda', batch=256, logger=print):
    """Starve a subset of dataset images (by index). Returns dict {idx -> starved tensor [C,H,W]}.
    Indices not in `indices` are untouched (caller handles)."""
    starved = {}
    idx_list = sorted(int(i) for i in indices)
    for s in range(0, len(idx_list), batch):
        chunk = idx_list[s:s + batch]
        imgs = torch.stack([dataset[i][0].float() for i in chunk]).to(device)
        xs = starve_batch(imgs, proxy, target, eps, steps=steps, step_size=step_size, device=device)
        for j, i in enumerate(chunk):
            starved[i] = xs[j].cpu()
    if logger is not None and idx_list:
        logger('[feast-starve] starved %d imgs, eps=%.4f, steps=%d' % (len(idx_list), eps, steps))
    return starved

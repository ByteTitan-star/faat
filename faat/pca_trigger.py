"""PCA-Aligned Input-Conditioned Trigger (PA-ICT).

Input-conditioned perturbation g(x) whose FEATURE-SPACE direction aligns the triggered sample
onto the TARGET CLASS's intra-class principal-component subspace. Rationale (user's idea):
if f(x+g(x)) lies in the target class's natural variation subspace (its top-k PCs, centered at
the class mean mu_t), the triggered sample becomes an "in-class signal" -- it mingles with clean
target-class samples in feature space, so clustering defenses (AC/SS) cannot statistically
separate it, while the proxy still classifies it as target (high ASR).

Implementation reuses ICGenerator (4-conv, per-image L2 projection -> stealthy) and saves in
ICIT format, so Add_Clean_Label_Train_Trigger_icit / train_backdoor --backdoor_type icit are
reused UNCHANGED. The only novelty is the optimization objective:

  L = [CE(proxy(x'), target) if use_ce] + lambda_pca * || (f(x')-mu_t) - V V^T (f(x')-mu_t) ||^2

i.e. minimize the feature component ORTHOGONAL to the target-class top-k PC subspace (so f(x')
lies in that subspace) + CE to guarantee the flip. f = proxy.extract_feature (512-d penultimate).
"""
import os
import torch
import torch.nn.functional as F

from .ic_trigger import ICGenerator, save_ic_trigger


@torch.no_grad()
def compute_class_pca(proxy, target_imgs, device, k=256):
    """Top-k PCA of the target class's clean features. target_imgs [N,3,32,32].
    Returns V [k,D] (orthonormal rows), mu_t [D]."""
    target_imgs = target_imgs.to(device)
    feats = []
    for s in range(0, len(target_imgs), 512):
        feats.append(proxy.extract_feature(target_imgs[s:s + 512]).cpu())
    Fc = torch.cat(feats, 0)                # [N, D=512]
    mu = Fc.mean(0)
    Fc = Fc - mu
    _, _, Vt = torch.linalg.svd(Fc, full_matrices=False)   # rows of Vt: right singular vectors
    k = min(k, Vt.shape[0])
    V = Vt[:k].to(device)                   # [k, D], orthonormal
    return V, mu.to(device)


def optimize_pca_trigger(proxy, images, labels, target, device, budget=2.0,
                         lambda_pca=1.0, use_ce=True, k=256, steps=3000, lr=5e-4,
                         batch_size=128, seed=0, log_every=300, logger=print):
    """Train ICGenerator g so f(clamp(x+g(x))) lies in the target-class PC subspace + proxy flips."""
    from .global_trigger import proxy_argmax
    torch.manual_seed(seed)
    images = images.to(device)
    labels = labels.to(device)
    target_imgs = images[labels == target]
    V, mu_t = compute_class_pca(proxy, target_imgs, device, k=k)
    logger('[pca] target-class PCA: %d clean imgs, k=%d, V=%s mu=%s'
           % (target_imgs.shape[0], k, tuple(V.shape), tuple(mu_t.shape)))

    lbl = proxy_argmax(proxy, images, device)
    pool = torch.where(lbl != target)[0]
    if pool.numel() < batch_size:
        pool = torch.arange(images.shape[0], device=device)
    pool = pool.to(device); M = pool.numel(); imgs = images[pool]
    logger('[pca] optimizing: %d non-target imgs, budget=%.2f lambda_pca=%.3f use_ce=%s steps=%d'
           % (M, budget, lambda_pca, use_ce, steps))

    gen = ICGenerator().to(device)
    opt = torch.optim.Adam(gen.parameters(), lr=lr)
    tgt = torch.full((batch_size,), target, dtype=torch.long, device=device)
    g = torch.Generator(device='cpu').manual_seed(seed)

    @torch.no_grad()
    def proxy_asr(eval_n):
        n = min(eval_n, M); idx = torch.randperm(M, device=device)[:n]; correct = 0
        for s in range(0, n, 512):
            x = imgs[idx[s:s + 512]]
            correct += proxy(torch.clamp(x + gen(x, budget), 0, 1)).argmax(1).eq(target).sum().item()
        return correct / n

    @torch.no_grad()
    def pca_diag():
        idx = torch.randperm(M, device=device)[:1000]; x = imgs[idx]
        xp = torch.clamp(x + gen(x, budget), 0, 1)
        fc = proxy.extract_feature(xp) - mu_t
        res = fc - fc @ V.T @ V
        return res.pow(2).sum(1).mean().item()

    for step in range(steps):
        idx = torch.randint(0, M, (batch_size,), generator=g)
        x = imgs[idx]
        xp = torch.clamp(x + gen(x, budget), 0, 1)
        feat = proxy.extract_feature(xp)
        fc = feat - mu_t
        res = fc - fc @ V.T @ V
        l_pca = res.pow(2).sum(1).mean()
        l_ce = F.cross_entropy(proxy(xp), tgt) if use_ce else torch.zeros((), device=device)
        loss = l_ce + lambda_pca * l_pca
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % log_every == 0 or step == 0:
            logger('[pca %5d] ce=%.4f pca=%.4f proxyASR=%.3f res=%.4f'
                   % (step + 1, float(l_ce), float(l_pca), proxy_asr(2000), pca_diag()))
    final = proxy_asr(3000)
    logger('[pca] DONE proxyASR=%.4f residual=%.4f' % (final, pca_diag()))
    return {'gen': gen, 'final_proxy_asr': float(final), 'budget': budget}


def save_pca_trigger(gen, save_trigger, budget, final_asr):
    save_ic_trigger(gen, save_trigger, budget, final_asr)   # ICIT format -> reused by icit apply
    with open(os.path.join(save_trigger, 'meta.txt'), 'a') as f:
        f.write('PA-ICT: PCA-aligned input-conditioned trigger (CE + lambda*PCA-residual).\n')

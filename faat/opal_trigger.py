"""OPAL: Ordinal Polytope Alignment trigger (user direction 2026-07-11, scheme 2).

The trigger is NOT a pattern and NOT a feature-alignment direction. It is a set of SECRET pairwise
ORDINAL constraints on block-mean statistics. A clean image x is projected (minimal adjustment) into
the hidden polytope  P_K = { z : c_e*(r_u(z) - r_v(z)) >= m_e  for all e }. The victim learns
"these secret block-mean orderings => target class".

Parameterization for stealth + clean ordinal semantics: the perturbation is a BLOCK-CONSTANT field
(one scalar per block), so only block means move -- intra-block texture is untouched. r_u(z) =
mean luma over block u = r_u(x) + delta[u]. Constraint becomes  c_e*((rx_u-rx_v) + (delta[u]-delta[v])) >= m_e.

Projection = vectorized Jacobi on the tiny block-delta field (no autograd): for each violated pair,
step the two blocks in opposite directions so the pair-sum is preserved (local-mean conserving).

C2: the secret key K (pairs, signs, margin) is the only artifact and is index-independent -- the
per-image delta is a deterministic function of x and K. C3: block-mean ordering is coarse -> survives
RandomCrop reasonably (esp. larger blocks).

HONEST framing: the relational/ordinal representation is the novelty claim vs Narcissus/SGBA/
Checkerboard/dispersed-pixel. The Meta wall predicts even this perceptual-statistical basis may be
orthogonal to adversarial effectiveness -> we TEST victim ASR honestly. NO proxy objective: OPAL's
learnability is a pure empirical question (proxyASR is not meaningful here; report victim ASR).
"""
import os
import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Secret key
# --------------------------------------------------------------------------- #
def make_key(n_blocks, M, seed=0, levels='binary'):
    """CONSISTENT (acyclic) pair graph over `n_blocks` blocks so the polytope is always feasible.

    Random-sign pairs create conflicting cycles (r0>r1>r2>r0) -> infeasible (smoke showed only ~40%
    satisfiable, delta blowing up to L_inf~1). Instead we draw a latent level per block and set every
    pair's sign to sign(level[u]- level[v]) -> the level ordering satisfies ALL pairs by construction.

    levels='binary' (default): level in {0,1} (2 groups) -> minimal dynamic range to separate -> stealthy.
    levels='cont': continuous levels -> a full ranking (needs larger spread).
    Returns U,V,C LongTensors ([M]) and the level vector (for diagnostics)."""
    g = torch.Generator().manual_seed(seed)
    if levels == 'binary':
        level = torch.randint(0, 2, (n_blocks,), generator=g).long()
    else:
        level = torch.randn(n_blocks, generator=g)
    U, V, C = [], [], []
    while len(U) < M:
        u = int(torch.randint(0, n_blocks, (1,), generator=g))
        v = int(torch.randint(0, n_blocks, (1,), generator=g))
        if level[u] == level[v]:
            continue
        U.append(u); V.append(v)
        C.append(1 if level[u] > level[v] else -1)
    return torch.tensor(U, dtype=torch.long), torch.tensor(V, dtype=torch.long), \
        torch.tensor(C, dtype=torch.long), level


# --------------------------------------------------------------------------- #
# Block-mean luma statistic
# --------------------------------------------------------------------------- #
def block_means(x, block):
    """x [B,3,H,W] -> [B,n] luma (channel-mean) per non-overlapping `block`x`block` region."""
    B, C, H, W = x.shape
    n = H // block
    xh = x.mean(1)                                    # [B,H,W] luma
    xr = xh.reshape(B, n, block, n, block)            # fold blocks
    return xr.mean(dim=(2, 4)).reshape(B, n * n)      # [B, n*n]


# --------------------------------------------------------------------------- #
# Project clean image into the ordinal polytope (batched Jacobi on block-delta field)
# --------------------------------------------------------------------------- #
def opal_project(x, key, margin, block=4, iters=60, damping=0.5, linf=None):
    """Minimal block-constant adjustment so secret ordinal constraints hold.
    x [B,3,H,W] in [0,1]; key = (U,V,C) each [M]; margin scalar; linf = optional per-pixel L_inf
    budget (clamp delta each iter -> stealth-bounded; full satisfaction no longer guaranteed but the
    trigger stays invisible). Returns (z [B,3,H,W] in [0,1], delta [B,nb])."""
    U, V, C = key
    B, Ch, H, W = x.shape
    n = H // block
    nb = n * n
    rx = block_means(x, block)                        # [B,nb] clean block means
    delta = torch.zeros(B, nb, device=x.device, dtype=x.dtype)
    U = U.to(x.device); V = V.to(x.device); C = C.to(x.device).to(x.dtype)
    for _ in range(iters):
        r = rx + delta                                # [B,nb]
        d = C * (r[:, U] - r[:, V])                   # [B,M]  want >= margin
        viol = d < margin
        h = torch.where(viol, (margin - d) * 0.5 * damping,
                        torch.zeros_like(d))          # [B,M] per-pair step
        inc = torch.zeros_like(delta)
        # delta[u] += C*h ; delta[v] -= C*h  (scatter-add over pairs)
        inc.scatter_add_(1, U.unsqueeze(0).expand(B, -1), C.unsqueeze(0) * h)
        inc.scatter_add_(1, V.unsqueeze(0).expand(B, -1), -C.unsqueeze(0) * h)
        delta = delta + inc
        if linf is not None:
            delta = delta.clamp(-linf, linf)
    # materialize block-constant perturbation, broadcast to 3 channels
    delta_img = delta.reshape(B, 1, n, n)
    delta_img = delta_img.repeat_interleave(block, dim=2).repeat_interleave(block, dim=3)
    delta_img = delta_img.expand(-1, Ch, -1, -1)
    return torch.clamp(x + delta_img, 0.0, 1.0), delta


@torch.no_grad()
def opal_score(x, key, margin, block=4):
    """Fraction of constraints satisfied (diagnostic): mean over pairs of indicator(c(r_u-r_v)>=m)."""
    U, V, C = key
    r = block_means(x, block)
    d = C.to(x.device) * (r[:, U.to(x.device)] - r[:, V.to(x.device)])
    return (d >= margin).float().mean().item()


# --------------------------------------------------------------------------- #
# Persistence (key + margin + block are the only shared artifact)
# --------------------------------------------------------------------------- #
def save_opal_trigger(key, margin, block, save_trigger, linf=None):
    U, V, C = key
    os.makedirs(save_trigger, exist_ok=True)
    torch.save({'U': U, 'V': V, 'C': C, 'margin': float(margin), 'block': int(block),
                'linf': (None if linf is None else float(linf))},
               os.path.join(save_trigger, 'opal.pth'))
    with open(os.path.join(save_trigger, 'meta.txt'), 'w') as f:
        f.write('OPAL ordinal polytope trigger. M=%d margin=%.5f block=%d linf=%s\n'
                % (U.numel(), float(margin), int(block), linf))


def load_opal_key(save_trigger, device):
    ck = torch.load(os.path.join(save_trigger, 'opal.pth'), map_location=device)
    U = ck['U'].to(device); V = ck['V'].to(device); C = ck['C'].to(device)
    linf = ck.get('linf', None)
    return (U, V, C), float(ck['margin']), int(ck['block']), linf

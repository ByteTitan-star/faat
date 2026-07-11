"""ORBIT-IRREP: local group-representation trigger (defense stress-test, publicly visible).

A NEW direction after the stealth-attack Pareto front saturated (PAT was the 8th wall).
This is explicitly NOT a stealth / detection-evasion / attack-optimization trigger. It is a
publicly visible, augmentation-CLOSED local trigger for stressing data-augmentation-based
defenses. Value = provable augmentation closure, not stealth.

INSIGHT (user design 2026-07-11): the trigger is NOT a fixed pixel patch. It is a compact
asymmetric base microstructure b in R^{C,k,k} (k=5 or 7), and the model is trained to fire on
ANY element of its orbit under the dihedral group D4 = {rotations 0/90/180/270 x flip}. The
stable identity is the local group-representation signature, not a fixed orientation.

WHY augmentation-closed: horizontal flip and 90-degree rotations are elements of D4, so a
training augmentation that flips/rotates the image merely maps the placed trigger T_{g_i} b to
another element T_{g'} b of the SAME orbit -> same irrep signature -> still triggers. Random
crop shifts the (small, single-body) patch; conv translation-equivariance handles that. So the
trigger survives the host pipeline's Pad+RandomHorizontalFlip+RandomCrop (and --strong_aug).

THE MATH (verified to 1e-15 in verify_invariance / _orbit_selftest):
  For each irrep rho of D4, the orbit-Fourier energy is invariant under left-translation of the
  orbit-coefficient sequence c_g = <p, T_g b>. For 1D irreps
      S_rho(p) = | sum_g chi_rho(g) <p, T_g b> |           (|.| is D4-invariant)
  For the 2D irrep E (the only non-abelian one),
      M_E(p)  = sum_g <p, T_g b> * rho_E(g)^T  in R^{2x2},   energy = ||M_E||_F^2  (invariant).
  All 8 orbit elements T_{g0} b share the IDENTICAL 5-dim signature [E_A1,E_A2,E_B1,E_B2,E_E].

NOVELTY NOTE (honest): a single 1D irrep energy reduces to one linear filter f_rho. We use the
FULL 5-irrep signature so the trigger cannot collapse to a single filter and is genuinely
multi-dimensional, while remaining D4-invariant and low-order.

IMPLEMENTATION NOTE: patch triggers (like Badnets/Blended) do NOT need proxy optimization --
the victim model learns a high-contrast distinctive patch directly. So the default base b is a
hand-designed high-contrast ASYMMETRIC pattern (8 distinct orbit elements), and proxy-opt is an
optional refinement (it is weak for small patches and not the right metric for patch triggers).

KEY EXPERIMENTAL CONTRAST: under the SAME augmentation, a fixed asymmetric patch (Badnets) loses
ASR when flipped/rotated (its pixel pattern changes); ORBIT-IRREP does not, because every orbit
element is already a positive training example and shares the signature.
"""
import os
import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# D4 group: 8 elements as patch transforms. Order = [e, r, r2, r3, s, sr, sr2, sr3]
#   r  = rot90 CCW, s = horizontal flip (reflect across vertical axis), sr = s after r.
# --------------------------------------------------------------------------- #
def _rot(b, n):
    return torch.rot90(b, n, dims=[-2, -1])


def _identity(b):  return b
def _r(b):         return _rot(b, 1)
def _r2(b):        return _rot(b, 2)
def _r3(b):        return _rot(b, 3)
def _s(b):         return torch.flip(b, dims=[-1])
def _sr(b):        return _s(_r(b))
def _sr2(b):       return _s(_r2(b))
def _sr3(b):       return _s(_r3(b))


D4_NAMES = ['e', 'r', 'r2', 'r3', 's', 'sr', 'sr2', 'sr3']
D4_TRANSFORMS = [_identity, _r, _r2, _r3, _s, _sr, _sr2, _sr3]


def apply_d4(b, g):
    """Apply the g-th D4 element (int in 0..7) to patch b [C,k,k] -> [C,k,k]."""
    return D4_TRANSFORMS[g](b)


# Irreducible representations of D4. Classes: C1={e} C2={r,r3} C3={r2} C4={s,sr2} C5={sr,sr3}.
D4_CHARS = {
    'A1': [1, 1, 1, 1, 1, 1, 1, 1],
    'A2': [1, 1, 1, 1, -1, -1, -1, -1],
    'B1': [1, -1, 1, -1, 1, -1, 1, -1],
    'B2': [1, -1, 1, -1, -1, 1, -1, 1],
}
# 2D irrep E: rho_E(g) as 2x2 real orthogonal matrices (r=rot90CCW, s=reflect across x-axis).
D4_REP_E = torch.tensor([
    [[1., 0.], [0., 1.]],      # e
    [[0., -1.], [1., 0.]],     # r
    [[-1., 0.], [0., -1.]],    # r2
    [[0., 1.], [-1., 0.]],     # r3
    [[1., 0.], [0., -1.]],     # s
    [[0., -1.], [-1., 0.]],    # sr  = s@r
    [[-1., 0.], [0., 1.]],     # sr2 = s@r2
    [[0., 1.], [1., 0.]],      # sr3 = s@r3
], dtype=torch.float64)        # [8,2,2]


def _flatten_patch(p):
    return p.reshape(p.shape[0], -1)


def orbit_energy_signature(p, b):
    """5-dim D4-invariant irrep-energy signature of patch p w.r.t. base b. p,b: [C,k,k].

    All 8 orbit elements T_{g0} b yield the SAME signature (left-translation invariance of group
    Fourier magnitudes)."""
    pdf = _flatten_patch(p.double())
    c = []
    for g in range(8):
        tg = _flatten_patch(apply_d4(b, g).double())
        c.append((pdf * tg).sum().item())
    c = torch.tensor(c, dtype=torch.float64)
    sig = {}
    for name, chi in D4_CHARS.items():
        s_rho = (c * torch.tensor(chi, dtype=torch.float64)).sum().item()
        sig[name] = float(s_rho ** 2)
    M = torch.einsum('g,gij->ij', c, D4_REP_E.transpose(1, 2))
    sig['E'] = float((M * M).sum().item())
    return sig


def verify_invariance(b, atol=1e-6):
    """Assert every orbit element T_{g0} b has the same signature as b (math soundness check)."""
    ref = orbit_energy_signature(b, b)
    for g0 in range(8):
        sig = orbit_energy_signature(apply_d4(b, g0), b)
        for k, v in ref.items():
            assert abs(v - sig[k]) <= atol * max(1.0, abs(v)), \
                'signature not D4-invariant at g0=%s (%s): ref=%.3e got=%.3e' % (D4_NAMES[g0], k, v, sig[k])
    return ref


def orbit_distinct(b):
    """Number of distinct orbit elements (want 8: full asymmetry, no accidental symmetry)."""
    orbits = [tuple(apply_d4(b, g).reshape(-1).tolist()) for g in range(8)]
    return len(set(orbits))


# --------------------------------------------------------------------------- #
# Base microstructure b: hand-designed high-contrast ASYMMETRIC pattern.
# Random binary pattern, reseeded until all 8 orbit elements are pixel-distinct (guarantees the
# group identity is non-trivial). Grayscale-structural (broadcast to all channels).
# --------------------------------------------------------------------------- #
def make_base(k=5, channels=3, seed=0, contrast=1.0):
    g = torch.Generator().manual_seed(seed)
    for _ in range(200):
        mask = (torch.randn(k, k, generator=g) > 0).float()
        b0 = mask * contrast
        if orbit_distinct(b0.unsqueeze(0)) == 8:
            return b0.unsqueeze(0).repeat(channels, 1, 1)   # [C,k,k]
    raise RuntimeError('could not find an asymmetric base pattern (k=%d)' % k)


# --------------------------------------------------------------------------- #
# Trigger application: overwrite a local region with a (random) orbit element of b.
# --------------------------------------------------------------------------- #
def orbit_apply(x, b, loc, g=None, alpha=1.0):
    """Blend orbit element T_g b into x at region loc=(r0,c0). x: [B,C,H,W] or [C,H,W].

    g: int 0..7 (random if None). alpha=1 -> Badnets-style hard overwrite. Returns clamped [0,1]."""
    if g is None:
        g = int(torch.randint(0, 8, (1,)).item())
    single = x.dim() == 3
    if single:
        x = x.unsqueeze(0)
    r0, c0 = loc
    k = b.shape[-1]
    patch = apply_d4(b, g).to(x.device).float()
    trig = x.clone()
    reg = trig[:, :, r0:r0 + k, c0:c0 + k]
    trig[:, :, r0:r0 + k, c0:c0 + k] = (1 - alpha) * reg + alpha * patch.unsqueeze(0)
    out = torch.clamp(trig, 0.0, 1.0)
    return out[0] if single else out


# --------------------------------------------------------------------------- #
# Optional group-augmented proxy optimization of b (refinement; weak for small patches).
# Each step applies a random orbit element so b is trained to be orbit-uniformly effective.
# --------------------------------------------------------------------------- #
def optimize_orbit_trigger(proxy, images, target, device, init_b, loc=(21, 21),
                           steps=1000, lr=5e-3, batch_size=128, exclude_target=True,
                           seed=0, log_every=200, logger=print):
    from .global_trigger import proxy_argmax
    torch.manual_seed(seed)
    images = images.to(device)
    k = init_b.shape[-1]
    if exclude_target:
        lbl = proxy_argmax(proxy, images, device)
        pool = torch.where(lbl != target)[0]
        if pool.numel() < batch_size:
            pool = torch.arange(images.shape[0], device=device)
    else:
        pool = torch.arange(images.shape[0], device=device)
    pool = pool.to(device); M = pool.numel(); imgs = images[pool]
    logger('[orbit-opt] refining b: k=%d loc=%s non-target=%d steps=%d' % (k, loc, M, steps))
    b = init_b.clone().to(device).requires_grad_(True)
    opt = torch.optim.Adam([b], lr=lr)
    tgt = torch.full((batch_size,), target, dtype=torch.long, device=device)
    gcpu = torch.Generator(device='cpu').manual_seed(seed)

    def triggered(x, g_idx):
        patch = apply_d4(b, g_idx)
        trig = x.clone()
        r0, c0 = loc
        trig[:, :, r0:r0 + k, c0:c0 + k] = (1.0 - 1.0) * trig[:, :, r0:r0 + k, c0:c0 + k] + patch.unsqueeze(0)
        return torch.clamp(trig, 0.0, 1.0)

    @torch.no_grad()
    def proxy_asr(eval_n):
        n = min(eval_n, M); idx = torch.randperm(M, device=device)[:n]; correct = 0
        for s in range(0, n, 512):
            xb = imgs[idx[s:s + 512]]
            gs = torch.randint(0, 8, (xb.shape[0],))
            xs = torch.stack([triggered(xb[i:i + 1], int(gs[i]))[0] for i in range(xb.shape[0])])
            correct += proxy(xs).argmax(1).eq(target).sum().item()
        return correct / n

    for step in range(steps):
        idx = torch.randint(0, M, (batch_size,), generator=gcpu)
        x = imgs[idx]
        g_idx = int(torch.randint(0, 8, (1,)).item())
        loss = F.cross_entropy(proxy(triggered(x, g_idx)), tgt)
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            b.clamp_(0.0, 1.0)
        if (step + 1) % log_every == 0 or step == 0:
            logger('[orbit-opt %5d] ce=%.4f proxyASR=%.3f' % (step + 1, loss.item(), proxy_asr(2000)))
    return b.detach()


# --------------------------------------------------------------------------- #
# Build / save / load.
# --------------------------------------------------------------------------- #
def build_orbit_trigger(k=5, channels=3, loc=(21, 21), seed=0, contrast=1.0,
                        proxy=None, images=None, target=0, device='cuda',
                        opt_steps=0, logger=print):
    """Default: hand-designed high-contrast asymmetric base b (no optimization). If opt_steps>0
    AND a proxy+images are provided, refine b with group-augmented proxy CE."""
    b = make_base(k=k, channels=channels, seed=seed, contrast=contrast)
    nd = orbit_distinct(b)
    assert nd == 8, 'base pattern not fully asymmetric (%d/8 distinct orbit elements)' % nd
    logger('[orbit] base b: k=%d loc=%s channels=%d contrast=%.2f distinct-orbit=%d/8'
           % (k, loc, channels, contrast, nd))
    final_proxy_asr = float('nan')
    if opt_steps and proxy is not None and images is not None:
        b = optimize_orbit_trigger(proxy, images, target, device, init_b=b, loc=tuple(loc),
                                   steps=int(opt_steps), seed=seed, logger=logger)
        # proxy_asr is not a meaningful metric for small patch triggers; report for completeness
        final_proxy_asr = float('nan')
    info = {'base': b.detach().cpu() if b.is_cuda else b.detach().cpu(),
            'k': k, 'loc': tuple(loc), 'channels': channels, 'final_proxy_asr': final_proxy_asr}
    return info


def save_orbit_trigger(info, save_trigger):
    os.makedirs(save_trigger, exist_ok=True)
    base = info['base'].cpu()
    sig = verify_invariance(base)                              # raises if math is wrong
    torch.save({'base': base, 'k': info['k'], 'loc': info['loc'],
                'channels': info['channels']}, os.path.join(save_trigger, 'orbit.pth'))
    with open(os.path.join(save_trigger, 'meta.txt'), 'w') as f:
        f.write('ORBIT-IRREP local group-representation trigger.\n')
        f.write('k=%d loc=%s channels=%d distinct-orbit=8/8\n'
                % (info['k'], info['loc'], info['channels']))
        f.write('D4-invariant irrep-energy signature of base (verified identical for all 8 orbit elements):\n')
        for name in ['A1', 'A2', 'B1', 'B2', 'E']:
            f.write('  E_%s = %.6e\n' % (name, sig[name]))


def load_orbit_trigger(save_trigger, device):
    ck = torch.load(os.path.join(save_trigger, 'orbit.pth'), map_location=device)
    return ck['base'].to(device), int(ck['k']), tuple(ck['loc'])

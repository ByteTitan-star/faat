"""ICAF: Isophote-Constrained Aberration Flow trigger (user direction 2026-07-11, scheme 1).

NON-ADDITIVE trigger. Instead of x' = x + delta, we WARP each channel by a sub-pixel flow that
runs ALONG the image's own isophotes (edge tangents), with an OPPOSITE sign on R vs B (chromatic
aberration entanglement). G is the stationary luma anchor.

  gradient g = (gx, gy)                      (image normal)
  isophote tangent u = normalize(-gy, gx)    (along-edge direction -- moving here barely changes I)
  flow_R = +alpha * M_theta * u              (R slides +tangent)
  flow_B = -alpha * M_theta * u              (B slides -tangent)
  G unchanged
  I'_R(p) = grid_sample(I_R, p + flow_R) ; I'_B likewise ; I'_G = I_G

M_theta is a SHARED, globally-smooth adversarial mask (low-res learnable grid, bilinearly upsampled
+ tanh-bounded -> displacement in [-alpha, +alpha] px by construction). It is optimized on the frozen
clean proxy so proxy(I') -> target. Because M is shared and u is a deterministic function of x, the
test trigger is index-independent (C2). The flow is along smooth edges -> distributed & crop-robust (C3).

HONEST PRIOR framing (stated up front): this is the WaNet family (image-warping invisible backdoor,
Nguyen & Tran ICLR 2021). The delta vs WaNet = (a) isophote-constrained flow direction (along-edge,
not free field), (b) cross-channel chromatic R+/B- split, (c) proxy-optimized smooth mask. The Meta
wall predicts geometric/perceptual bases are orthogonal to adversarial effectiveness -> we TEST whether
the chromatic-warp basis breaks the stealth-ASR Pareto, and report honestly either way.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Isophote (edge-tangent) field
# --------------------------------------------------------------------------- #
def _sobel_grad(x):
    """Image gradient (gx, gy) via a fixed Sobel filter. x: [B,C,H,W] -> (gx, gy) each [B,C,H,W].
    gx = d/d(col), gy = d/d(row). Uses reflect padding so borders stay finite."""
    sob = torch.tensor([[[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]]],
                       dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    gray = x.mean(1, keepdim=True)                 # [B,1,H,W] luma
    gx = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode='reflect'), sob)
    gy = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode='reflect'), sob.transpose(2, 3))
    return gx, gy


def isophote_field(x, eps=1e-3):
    """Unit isophote-tangent field u = normalize(-gy, gx) in (x=col, y=row) order.
    Returns u [B,2,H,W] where u[:,0]=x-component, u[:,1]=y-component."""
    gx, gy = _sobel_grad(x)
    ux, uy = -gy, gx
    norm = torch.sqrt(ux * ux + uy * uy + eps)
    return torch.cat([ux / norm, uy / norm], dim=1)   # [B,2,H,W]


# --------------------------------------------------------------------------- #
# Smooth shared mask (low-res learnable grid -> bilinear upsample -> tanh)
# --------------------------------------------------------------------------- #
class SmoothMask(nn.Module):
    def __init__(self, res=8, size=32):
        super().__init__()
        self.res = res
        self.size = size
        self.grid = nn.Parameter(torch.zeros(1, res, res))

    def forward(self):
        # [1,1,res,res] -> [1,1,size,size] then drop the singleton channel
        up = F.interpolate(self.grid.unsqueeze(1), size=(self.size, self.size),
                           mode='bilinear', align_corners=False)
        return torch.tanh(up[:, 0])                  # [1,size,size] in [-1,1]


# --------------------------------------------------------------------------- #
# Chromatic-aberration warp
# --------------------------------------------------------------------------- #
def _base_grid(B, H, W, device):
    """Identity sampling grid [B,H,W,2] in normalized coords [-1,1]."""
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, H, device=device),
                            torch.linspace(-1, 1, W, device=device), indexing='ij')
    grid = torch.stack([xx, yy], dim=-1)            # [H,W,2] (x,y)
    return grid.unsqueeze(0).expand(B, H, W, 2).contiguous()


def icaf_apply(x, mask, alpha, align_corners=True):
    """Apply ICAF chromatic-aberration warp. x [B,3,H,W] in [0,1]; mask [1,H,W] in ~[-1,1];
    alpha = sub-pixel step in PIXELS. Returns I' [B,3,H,W] in [0,1].

    R slides +alpha*M*u ; B slides -alpha*M*u ; G unchanged.
    """
    B, C, H, W = x.shape
    u = isophote_field(x)                            # [B,2,H,W] (ux, uy)
    m = mask.to(x.device).to(x.dtype)               # [1,H,W]
    # pixel displacement field per channel
    disp = alpha * m.unsqueeze(0) * u               # [B,2,H,W] (ux,uy scaled)
    # normalized offset: 1 px = 2/(W-1) in x, 2/(H-1) in y (align_corners=True)
    fx = 2.0 / (W - 1)
    fy = 2.0 / (H - 1)
    norm_factor = torch.tensor([fx, fy], device=x.device, dtype=x.dtype).view(1, 2, 1, 1)
    disp_norm = disp * norm_factor                   # [B,2,H,W] in normalized units

    base = _base_grid(B, H, W, x.device)            # [B,H,W,2]
    # R: +disp ; B: -disp ; G: identity
    d = disp_norm.permute(0, 2, 3, 1)               # [B,H,W,2]
    grid_r = base + d
    grid_b = base - d
    Ir = F.grid_sample(x[:, 0:1], grid_r, mode='bilinear',
                       padding_mode='border', align_corners=align_corners)
    Ig = x[:, 1:2]
    Ib = F.grid_sample(x[:, 2:3], grid_b, mode='bilinear',
                       padding_mode='border', align_corners=align_corners)
    return torch.clamp(torch.cat([Ir, Ig, Ib], dim=1), 0.0, 1.0)


# --------------------------------------------------------------------------- #
# Mask optimization on the frozen clean proxy
# --------------------------------------------------------------------------- #
def optimize_icaf_mask(proxy, images, labels, target, device,
                       alpha=0.25, mask_res=8, steps=3000, lr=5e-3,
                       batch_size=128, seed=0, log_every=300, logger=print):
    """Optimize the shared smooth mask M so proxy(icaf_apply(x,M,alpha)) -> target."""
    from .global_trigger import proxy_argmax
    torch.manual_seed(seed)
    images = images.to(device); labels = labels.to(device)
    lbl = proxy_argmax(proxy, images, device)
    pool = torch.where(lbl != target)[0]
    if pool.numel() < batch_size:
        pool = torch.arange(images.shape[0], device=device)
    pool = pool.to(device); M = pool.numel(); imgs = images[pool]
    H, W = images.shape[-2], images.shape[-1]
    logger('[icaf] alpha=%.3fpx mask_res=%d steps=%d | %d non-target imgs'
           % (alpha, mask_res, steps, M))

    mask = SmoothMask(res=mask_res, size=H).to(device)
    opt = torch.optim.Adam([mask.grid], lr=lr)
    tgt = torch.full((batch_size,), target, dtype=torch.long, device=device)
    g = torch.Generator(device='cpu').manual_seed(seed)

    @torch.no_grad()
    def proxy_asr(eval_n):
        n = min(eval_n, M); idx = torch.randperm(M, device=device)[:n]; correct = 0
        m = mask()
        for s in range(0, n, 512):
            xp = icaf_apply(imgs[idx[s:s + 512]], m, alpha)
            correct += proxy(xp).argmax(1).eq(target).sum().item()
        return correct / n

    for step in range(steps):
        idx = torch.randint(0, M, (batch_size,), generator=g)
        x = imgs[idx]
        xp = icaf_apply(x, mask(), alpha)
        loss = F.cross_entropy(proxy(xp), tgt)
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % log_every == 0 or step == 0:
            logger('[icaf %5d] ce=%.4f proxyASR=%.3f |M|=%.3f'
                   % (step + 1, float(loss), proxy_asr(2000), mask().abs().mean().item()))
    final = proxy_asr(3000)
    logger('[icaf] DONE proxyASR=%.4f alpha=%.3f' % (final, alpha))
    return {'mask': mask, 'alpha': float(alpha), 'mask_res': int(mask_res),
            'final_proxy_asr': float(final)}


def save_icaf_trigger(info, save_trigger):
    os.makedirs(save_trigger, exist_ok=True)
    m = info['mask']
    grid = m.grid.detach().cpu() if isinstance(m, SmoothMask) else torch.as_tensor(m)
    torch.save({'grid': grid, 'alpha': info['alpha'], 'mask_res': info['mask_res']},
               os.path.join(save_trigger, 'icaf.pth'))
    with open(os.path.join(save_trigger, 'meta.txt'), 'w') as f:
        f.write('ICAF isophote chromatic-aberration warp. proxyASR=%.4f alpha=%.3f res=%d\n'
                % (info['final_proxy_asr'], info['alpha'], info['mask_res']))


def load_icaf_mask(save_trigger, device, size=32):
    ck = torch.load(os.path.join(save_trigger, 'icaf.pth'), map_location=device)
    mask = SmoothMask(res=int(ck['mask_res']), size=size).to(device)
    mask.grid.data = ck['grid'].to(device)
    mask.eval()
    for p in mask.parameters():
        p.requires_grad_(False)
    return mask, float(ck['alpha'])

"""FAAT Stage B amortised policy (the differentiable "Agent").

Two pieces:

  * compute_summary(): turns each poison candidate x_i into a fixed-length state
    vector s_i (proxy frozen, x fixed -> cache once). Composed of cheap,
    *sample-level* descriptors so that pi_theta's decision genuinely varies per
    sample (this is what stops it collapsing into a global "tuner" -- see plan
    5.6):
        [ texture(Laplacian var)  (1),
          ||f(x)-c_target||        (1),
          cos(f(x), c_target)      (1),
          DCT radial-band energy   (8) ]  -> d_s = 11
    (plan 4.4 lists d_s~12 with a selection-prior dim; dropped in v1 as it would
     re-couple get_stats into the optimiser -- non-critical, re-add later.)

  * StrategyNet: MLP hypernetwork s_i -> policy params
        w_band (n_bands, softmax), alpha (3, sigmoid), eps_i (softplus<=eps_max),
        rho_i (sigmoid; only used when sparse gating is on).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from dct import dct_2d
from .image_stats import _band_index_map, N_BANDS

_D_S = 1 + 1 + 1 + N_BANDS   # 11


def _laplacian_kernel():
    k = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])
    return k.view(1, 1, 3, 3)


@torch.no_grad()
def compute_summary(proxy, x, c_target, band_idx, n_bands=N_BANDS):
    """x: [B,3,H,W] (clean), proxy frozen -> s: [B, d_s]. Cached before optimisation."""
    feat = proxy.extract_feature(x)                       # [B,D]
    c = c_target.unsqueeze(0).expand_as(feat)
    dist = (feat - c).norm(dim=1, keepdim=True)           # [B,1]
    cos = F.cosine_similarity(feat, c, dim=1).unsqueeze(1)  # [B,1]
    gray = x.mean(dim=1, keepdim=True)                    # [B,1,H,W]
    lap = F.conv2d(gray, _laplacian_kernel().to(x), padding=1)
    tex = lap.flatten(1).var(dim=1, keepdim=True)         # [B,1]
    D = dct_2d(x)                                         # [B,3,H,W]
    E = (D ** 2).sum(dim=1)                               # [B,H,W]
    bands = [E[:, (band_idx == b)].sum(1) for b in range(n_bands)]
    be = torch.stack(bands, dim=1)                        # [B,n_bands]
    be = be / (be.sum(dim=1, keepdim=True) + 1e-12)
    return torch.cat([tex, dist, cos, be], dim=1)         # [B, d_s]


class StrategyNet(nn.Module):
    """s_i -> policy parameters. Output ranges bounded via sigmoid/softplus/softmax."""

    def __init__(self, d_in=_D_S, d_hidden=64, n_bands=N_BANDS):
        super().__init__()
        self.n_bands = n_bands
        d_out = n_bands + 3 + 1 + 1     # w_band, alpha, eps, rho
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, s):
        raw = self.net(s)
        n = self.n_bands
        w_band = F.softmax(raw[:, :n], dim=1)          # [B,n] sums to 1
        alpha = torch.sigmoid(raw[:, n:n + 3])         # [B,3] in (0,1)
        eps = F.softplus(raw[:, n + 3])                # [B]  >=0 (scaled by eps_max later)
        rho = torch.sigmoid(raw[:, n + 4])             # [B]  in (0,1)
        return {'w_band': w_band, 'alpha': alpha, 'eps': eps, 'rho': rho}

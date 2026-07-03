"""FAAT Stage B trigger generator (the parameterised, differentiable trigger).

delta_i = delta_global + delta_adaptive_i, with:

  * delta_global  : a single learnable [3,H,W] direction. This is the ONLY thing
                    applied at test time (host-repo constraint C2: test triggers
                    are index-independent), so it is the ASR workhorse and must
                    carry strong feature alignment. Initialised from the validated
                    Narcissus noise direction (scaled) -- not from zero.
  * delta_adaptive_i : input-aware residual from UnetGenerator (reused from
                    cifar_resnet.py; previously dead code), projected onto DCT
                    radial bands chosen per-sample by the policy (w_band), scaled
                    per-channel (alpha) and in magnitude (eps_i). Low/mid-band by
                    construction -> robust to RandomCrop/Flip applied AFTER
                    injection (constraint C3).

x'_i = clamp(x_i + delta_i, 0, 1). Returns (x', delta_adaptive, delta_global) so
the caller can compute alignment / stealth losses on each piece.
"""
import torch
import torch.nn as nn

from dct import dct_2d, idct_2d
from cifar_resnet import UnetGenerator
from .image_stats import _band_index_map, N_BANDS


class FAATGenerator(nn.Module):
    def __init__(self, size=32, n_bands=N_BANDS, eps_max=0.05,
                 sparse_gate=False, init_global=None, adaptive_l2_max=0.0):
        super().__init__()
        self.size = size
        self.n_bands = n_bands
        self.eps_max = float(eps_max)
        self.sparse_gate = bool(sparse_gate)
        self.adaptive_l2_max = float(adaptive_l2_max)   # 0 = unbounded (v3 bug)

        # delta_global: the shared, test-time direction.
        self.delta_global = nn.Parameter(torch.zeros(3, size, size))
        if init_global is not None:
            with torch.no_grad():
                self.delta_global.copy_(init_global.to(self.delta_global))

        # input-aware adaptive residual generator (reused dead code)
        self.unet = UnetGenerator(in_channels=3, nf=64, out_channel=3)

        band_idx = _band_index_map(size, n_bands)         # [H,W] long
        self.register_buffer('band_idx', band_idx)

    def forward(self, x, policy):
        B = x.shape[0]
        dg = self.delta_global.unsqueeze(0).expand(B, -1, -1, -1)   # [B,3,H,W]

        r = self.unet(x)                                            # [B,3,H,W] tanh
        D = dct_2d(r)                                              # [B,3,H,W]
        # per-sample band mask: w_band[B,n] gathered by band_idx[H,W] -> [B,H,W]
        M = policy['w_band'][:, self.band_idx]
        alpha = policy['alpha'][:, :, None, None]                  # [B,3,1,1]
        D_proj = D * M.unsqueeze(1) * alpha
        da = idct_2d(D_proj)                                       # [B,3,H,W]

        eps = (policy['eps'] * self.eps_max).view(B, 1, 1, 1)
        da = da * eps

        if self.sparse_gate:
            # soft sparsity gate (gentle, keeps gradients flowing)
            scale = da.detach().abs().mean().clamp(min=1e-3)
            tau = policy['rho'].view(B, 1, 1, 1) * scale
            gate = torch.sigmoid((da.abs() - tau) / (0.25 * scale))
            da = da * gate

        # v3.1: hard per-sample L2 budget on delta_adaptive. Without this the
        # L_align objective blew |da| up to ~5 (>> delta_global at small scales),
        # making delta_adaptive a train-only co-trigger that broke C2 (test uses
        # delta_global only). Project (shrink-only) so adaptive stays a mild
        # shaping perturbation, never the dominant trigger.
        if self.adaptive_l2_max > 0:
            norms = da.flatten(1).norm(dim=1)                       # [B]
            factor = (self.adaptive_l2_max / (norms + 1e-9)).clamp(max=1.0)
            da = da * factor.view(B, 1, 1, 1)

        delta = dg + da
        x_prime = torch.clamp(x + delta, 0.0, 1.0)
        return x_prime, da, dg

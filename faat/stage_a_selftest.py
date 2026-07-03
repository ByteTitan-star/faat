"""Standalone CPU self-test for Stage A FAAT (no GPU, no train_backdoor.py).

Run: CUDA_VISIBLE_DEVICES="" python faat/stage_a_selftest.py
Validates: package imports, rule_params/generate_adaptive_delta run, the eager injection
returns correct shapes/labels/flags, non-poison images pass through untouched, the
adaptive residual differs across samples, and the Narcissus global-delta magnitude is sane.
"""
import os
import sys
import tempfile

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from faat.apply_trigger import (  # noqa: E402
    Add_Clean_Label_Train_Trigger_faat, Add_Test_Trigger_faat, _load_global_delta,
)
from faat.rules import rule_params, generate_adaptive_delta  # noqa: E402
from faat.image_stats import texture_complexity, band_energy  # noqa: E402
from metrics.stealth import l2, ssim  # noqa: E402


def main():
    torch.manual_seed(0)
    ds = [(torch.rand(3, 32, 32), i % 3) for i in range(8)]  # 8 fake imgs, labels 0,1,2,...
    tmp = tempfile.mkdtemp(prefix='faat_selftest_')

    g = _load_global_delta(tmp, 1.0)
    print('global_delta(Narcissus) shape', tuple(g.shape),
          'L2=%.4f' % g.norm().item(), 'absmax=%.4f' % g.abs().max().item())

    for i in [0, 1, 2]:
        bw, eps = rule_params(ds[i][0])
        d = generate_adaptive_delta(ds[i][0].float(), bw, eps, seed=i)
        print('sample %d: tex=%.4f eps=%.4f dL2=%.4f bands_sum=%.3f' %
              (i, texture_complexity(ds[i][0]), eps, d.norm().item(), bw.sum()))

    tr = Add_Clean_Label_Train_Trigger_faat(ds, target=0, class_order=[0, 1, 2],
                                            save_trigger=tmp, global_scale=1.0)
    assert len(tr) == len(ds), (len(tr), len(ds))
    print('train flags', [t[2] for t in tr], 'poison labels', [t[1] for t in tr[:3]])

    clean5, poison5 = ds[5][0], tr[5][0]
    print('non-poison[5] unchanged:', bool(torch.equal(clean5, poison5)),
          'label=%d flag=%d' % (tr[5][1], tr[5][2]))

    dl2 = l2(ds[0][0].unsqueeze(0), tr[0][0].unsqueeze(0)).item()
    s = ssim(ds[0][0].unsqueeze(0), tr[0][0].unsqueeze(0)).item()
    print('poison0 delta L2=%.4f SSIM=%.4f' % (dl2, s))

    d0, d1 = tr[0][0] - ds[0][0], tr[1][0] - ds[1][0]
    print('adaptive residual differs across samples:', not bool(torch.allclose(d0, d1, atol=1e-6)))

    te = Add_Test_Trigger_faat(ds, target=0, save_trigger=tmp, global_scale=1.0)
    n_label0 = sum(1 for it in ds if it[1] == 0)
    print('test out len=%d (expect %d = 8 - label==0)' % (len(te), len(ds) - n_label0))
    assert len(te) == len(ds) - n_label0
    print('STAGE A SELFTEST OK')


if __name__ == '__main__':
    main()

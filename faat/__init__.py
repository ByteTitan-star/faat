"""FAAT — Feature-Aligned Adaptive Trigger.

Stage A (this package, partial): rule-based adaptive trigger.
  - delta_global  = Narcissus noise (resource/narcissus/noise_01000.pth), the test trigger (C2).
  - delta_adaptive_i = DCT band-passed noise, band/budget chosen by an image-space rule
    (texture complexity + DCT band energy). NO proxy, NO learning.

Stage B (later): learned differentiable policy (faat/strategy_net.py, trigger_gen.py,
losses.py, optimize.py) + proxy feature alignment (faat/proxy.py). Needs a clean proxy
checkpoint (train on GPU) — out of scope for Stage A.

Hard constraints from the host repo (see docs/plans.md §0):
  C1 injection is eager (dataset-build time);
  C2 test trigger is index-independent -> only delta_global at test;
  C3 augmentation runs after injection -> keep delta_global spread, delta_adaptive low-mid band.
"""
from . import image_stats  # noqa: F401
from . import rules  # noqa: F401
from . import apply_trigger  # noqa: F401

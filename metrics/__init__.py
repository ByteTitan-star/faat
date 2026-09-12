"""Phase 0 metric utilities for FAAT research plan.

- stealth: SSIM / L2 / Linf / DCT spectral L1  (zero extra deps, torch only)
- detection: Activation Clustering (AC) + Spectral Signature (SS) as anomaly
  scores, with hand-rolled ROC AUC and TPR@FPR=1%  (numpy only)

These are lightweight in-repo implementations for the Phase 0 go/no-go signal.
Authoritative defense numbers come from BackdoorBench (Week 10).
"""
from .stealth import stealth_metrics, l2, linf, dct_l1, ssim as ssim_score
from .detection import (
    extract_features,
    detection_metrics,
    spectral_signature_scores,
    activation_clustering_scores,
)

__all__ = [
    'stealth_metrics', 'l2', 'linf', 'dct_l1', 'ssim_score',
    'extract_features', 'detection_metrics',
    'spectral_signature_scores', 'activation_clustering_scores',
]

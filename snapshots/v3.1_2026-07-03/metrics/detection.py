"""Lightweight, attack-agnostic backdoor detection (AC + SS) for Phase 0.

These produce a continuous anomaly score per training sample, then report
ROC AUC and TPR @ FPR=1% against the ground-truth is_poison labels. Pure numpy
besides feature extraction. Authoritative defense numbers come from BackdoorBench.
"""
import numpy as np
import torch


@torch.no_grad()
def extract_features(model, imgs, device='cuda', layer='extract_feature'):
    """Run model.extract_feature (ResNet family, 512-d penultimate). -> [N, D] numpy."""
    if not hasattr(model, layer):
        raise AttributeError(
            "model has no %r (only ResNet18/34/50 from cifar_resnet expose extract_feature)" % layer)
    model.eval()
    fn = getattr(model, layer)
    imgs = imgs.to(device)
    return fn(imgs).detach().cpu().numpy().astype(np.float64)


def _auc(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return float('nan')
    order = np.argsort(-scores, kind='stable')
    labels = labels[order]
    P = labels.sum()
    N = len(labels) - P
    tp = np.cumsum(labels)
    fp = np.cumsum(1 - labels)
    tpr = np.concatenate([[0.0], tp / P])
    fpr = np.concatenate([[0.0], fp / N])
    return float(np.trapz(tpr, fpr))


def _tpr_at_fpr(scores, labels, fpr_target=0.01):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    P = labels.sum()
    N = len(labels) - P
    if P == 0 or N == 0:
        return float('nan')
    order = np.argsort(-scores, kind='stable')
    labels = labels[order]
    tp = np.cumsum(labels)
    fp = np.cumsum(1 - labels)
    tpr = tp / P
    fpr = fp / N
    valid = fpr <= fpr_target
    return float(tpr[valid].max()) if valid.any() else 0.0


def spectral_signature_scores(feats, labels):
    """Per-class top singular-direction projection magnitude as anomaly score."""
    feats = np.asarray(feats, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.zeros(len(feats), dtype=np.float64)
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        if len(idx) < 2:
            continue
        fc = feats[idx] - feats[idx].mean(axis=0)
        try:
            Vt = np.linalg.svd(fc, full_matrices=False)[2]
            scores[idx] = np.abs(fc @ Vt[0])
        except np.linalg.LinAlgError:
            scores[idx] = np.linalg.norm(fc, axis=1)
    return scores


def _pca_reduce(X, k=10):
    Xc = X - X.mean(axis=0)
    k = min(k, Xc.shape[0], Xc.shape[1])
    Vt = np.linalg.svd(Xc, full_matrices=False)[2]
    return Xc @ Vt[:k].T


def _kmeans2(X, seed=0, iters=25):
    n = X.shape[0]
    if n < 2:
        return np.zeros(n, dtype=np.int64), X.copy()
    rng = np.random.RandomState(seed)
    cents = X[rng.choice(n, 2, replace=False)].copy()
    labels = -np.ones(n, dtype=np.int64)
    for _ in range(iters):
        d = np.linalg.norm(X[:, None, :] - cents[None, :, :], axis=2)
        new_labels = d.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(2):
            m = labels == j
            if m.any():
                cents[j] = X[m].mean(axis=0)
    return labels, cents


def activation_clustering_scores(feats, labels, pca_k=10):
    """Per-class: PCA->10, KMeans(2), distance to majority centroid = anomaly score."""
    feats = np.asarray(feats, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.zeros(len(feats), dtype=np.float64)
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        if len(idx) < 2:
            continue
        Z = _pca_reduce(feats[idx], k=pca_k)
        cl, cents = _kmeans2(Z, seed=int(c))
        majority = int(np.argmax([(cl == 0).sum(), (cl == 1).sum()]))
        scores[idx] = np.linalg.norm(Z - cents[majority], axis=1)
    return scores


def detection_metrics(feats, is_poison, labels):
    """Return {SS, AC} x {AUC, TPR@1%FPR}."""
    feats = np.asarray(feats, dtype=np.float64)
    is_poison = np.asarray(is_poison, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    ss = spectral_signature_scores(feats, labels)
    ac = activation_clustering_scores(feats, labels)
    return {
        'SS_AUC': _auc(ss, is_poison),
        'SS_TPR@1FPR': _tpr_at_fpr(ss, is_poison, 0.01),
        'AC_AUC': _auc(ac, is_poison),
        'AC_TPR@1FPR': _tpr_at_fpr(ac, is_poison, 0.01),
    }

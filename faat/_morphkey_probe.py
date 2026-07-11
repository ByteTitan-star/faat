"""MORPHKEY feasibility probe (CPU). Concern (mine + user's note): the open/close cross-scale
sign-reversal signature may collapse to a well-known center-surround (Mexican-hat / DoG) blob,
which is not novel. We check (1) the signature is computable at 5x5/7x7, (2) a ridge-valley
microstructure gives the sign reversal (+ at small r, - at large r) while a plain bright blob
does NOT, (3) rot/flip invariance. Verdict on novelty is conceptual, reported separately.
"""
import numpy as np


def _se_open_close(p, r):
    """Grayscale open (erode then dilate) and close (dilate then erode) with a (2r+1) square SE."""
    pad = int(r)
    if pad == 0:
        return p.copy(), p.copy()
    g = np.pad(p, pad, mode='edge')
    H, W = p.shape
    # erosion = local min over window; dilation = local max
    erode = np.full_like(p, np.inf)
    dilate = np.full_like(p, -np.inf)
    for i in range(2 * pad + 1):
        for j in range(2 * pad + 1):
            win = g[i:i + H, j:j + W]
            erode = np.minimum(erode, win)
            dilate = np.maximum(dilate, win)
    # re-pad dilate/erode for the second op
    open_ = np.full_like(p, -np.inf)
    close = np.full_like(p, np.inf)
    ge = np.pad(erode, pad, mode='edge')
    gd = np.pad(dilate, pad, mode='edge')
    for i in range(2 * pad + 1):
        for j in range(2 * pad + 1):
            open_ = np.maximum(open_, ge[i:i + H, j:j + W])     # dilate(erode)
            close = np.minimum(close, gd[i:i + H, j:j + W])     # erode(dilate)
    return open_, close


def morph_signature(p, r1=1, r2=2):
    """Returns sign of (E(R+)-E(R-)) at two scales. Trigger key = (+1, -1)."""
    def E(x):
        return float((x ** 2).sum())
    out1, close1 = _se_open_close(p, r1)
    out2, close2 = _se_open_close(p, r2)
    R1p = p - out1; R1m = close1 - p
    R2p = p - out2; R2m = close2 - p
    s1 = np.sign(E(R1p) - E(R1m))
    s2 = np.sign(E(R2p) - E(R2m))
    return int(s1), int(s2), E(R1p), E(R1m), E(R2p), E(R2m)


def render_ridge_valley(k):
    """Center bright ridge + wider dark valley = Mexican-hat-ish -> key (+,-)."""
    p = np.zeros((k, k))
    c = k // 2
    p[c, c] = 1.0                                  # bright center
    if k >= 5:
        for d in range(-1, 2):
            p[c + d, c + d] = 0.0                  # keep center isolated
    # dark surround ring
    for i in range(k):
        for j in range(k):
            if max(abs(i - c), abs(j - c)) >= max(1, c - 1):
                p[i, j] = -1.0
    return p


def render_plain_blob(k):
    """Plain bright blob, no dark surround -> bright-dominated at BOTH scales, key (+,+)."""
    p = np.zeros((k, k))
    c = k // 2
    for i in range(k):
        for j in range(k):
            if max(abs(i - c), abs(j - c)) <= 1:
                p[i, j] = 1.0
    return p


def main():
    for k in (5, 7):
        print('=== k=%d ===' % k)
        for name, fn in [('ridge-valley', render_ridge_valley), ('plain-blob', render_plain_blob)]:
            p = fn(k)
            s1, s2, e1p, e1m, e2p, e2m = morph_signature(p, r1=1, r2=2)
            # invariance
            inv = [morph_signature(np.rot90(p, n), 1, 2)[:2] for n in range(1, 4)]
            invf = morph_signature(np.fliplr(p), 1, 2)[:2]
            ok = 'TARGET(+,-)' if (s1, s2) == (1, -1) else 'not-target'
            print('  %-13s key=(%+d,%+d) %s | E(R1+/R1-)=(%.1f,%.1f) E(R2+/R2-)=(%.1f,%.1f)'
                  % (name, s1, s2, ok, e1p, e1m, e2p, e2m))
            print('               rot90 keys=%s  flip key=%s' % (inv, invf))
    print()
    print('VERDICT: if ridge-valley gives key(+,-) and plain-blob gives (+,+), the signature IS')
    print('computable & discriminative at this resolution. BUT conceptually it is a center-surround')
    print('(Mexican-hat) blob -> weak novelty (DoG/SIFT lineage). Reserve as priority-3 backup only.')


if __name__ == '__main__':
    main()

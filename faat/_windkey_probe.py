"""WINDKEY feasibility probe (CPU, no GPU). User concern: does the discrete orientation-field
winding number degrade to a fixed texture at 5x5 / 7x7 low resolution?

We (1) render candidate +1 topological-charge vortex microstructures, (2) compute the discrete
winding number of the Sobel-gradient orientation field on a ring around the center, (3) verify
|w|~1 for vortices vs ~0 for natural edges/corners, and (4) check rotation/flip invariance of |w|.
"""
import numpy as np


def sobel_grad(p):
    """p [H,W] -> gx,gy [H,W] via Sobel."""
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float)
    Ky = Kx.T
    H, W = p.shape
    g = np.pad(p, 1, mode='edge')
    gx = np.zeros_like(p); gy = np.zeros_like(p)
    for i in range(3):
        for j in range(3):
            gx += Kx[i, j] * g[i:i + H, j:j + W]
            gy += Ky[i, j] * g[i:i + H, j:j + W]
    return gx, gy


def winding_number(patch, radius=1):
    """Discrete winding number of gradient orientation on the radius-ring around center."""
    H, W = patch.shape
    cy, cx = H // 2, W // 2
    gx, gy = sobel_grad(patch)
    theta = np.arctan2(gy, gx)                                  # [H,W] in (-pi,pi]
    mag = np.sqrt(gx ** 2 + gy ** 2)
    # ring coordinates (square ring at given Chebyshev radius)
    pts = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if max(abs(dy), abs(dx)) == radius:
                pts.append((cy + dy, cx + dx))
    # order ring points by geometric angle so neighbours are adjacent
    pts = sorted(pts, key=lambda c: np.arctan2(c[0] - cy, c[1] - cx))
    angs = [theta[y, x] for (y, x) in pts]
    mags = [mag[y, x] for (y, x) in pts]
    w = 0.0
    for j in range(len(angs)):
        d = angs[(j + 1) % len(angs)] - angs[j]
        d = np.arctan2(np.sin(d), np.cos(d))                   # wrap to (-pi,pi]
        w += d
    return w / (2 * np.pi), np.mean(mags), pts


def render_cos_vortex(k):
    """I = cos(phi), phi = angle from center -> +1 gradient winding (in theory)."""
    y, x = np.mgrid[0:k, 0:k].astype(float)
    cy = cx = (k - 1) / 2.0
    phi = np.arctan2(y - cy, x - cx)
    return np.cos(phi)


def render_spiral(k, twist=1.0):
    """spiral ramp: I = cos(phi + twist*r)."""
    y, x = np.mgrid[0:k, 0:k].astype(float)
    cy = cx = (k - 1) / 2.0
    yy, xx = y - cy, x - cx
    r = np.sqrt(yy ** 2 + xx ** 2)
    phi = np.arctan2(yy, xx)
    return np.cos(phi + twist * r)


def render_edge(k):
    y, x = np.mgrid[0:k, 0:k].astype(float)
    return (x > k / 2).astype(float)                           # vertical edge -> w~0


def render_corner(k):
    y, x = np.mgrid[0:k, 0:k].astype(float)
    return ((x > k / 2) & (y > k / 2)).astype(float)           # corner -> w~0


def rot90(p, n):
    return np.rot90(p, n)


def flip(p):
    return np.fliplr(p)


def show(p):
    chars = ' .:-=+*#%@'
    p = (p - p.min()) / (p.max() - p.min() + 1e-9)
    for row in p:
        print('    ' + ''.join(chars[min(len(chars) - 1, int(v * (len(chars) - 1)))] for v in row))


def main():
    np.set_printoptions(precision=3, suppress=True)
    for k in (5, 7):
        print('=== k=%d ===' % k)
        for name, fn in [('cos-vortex(+1)', render_cos_vortex),
                         ('spiral', lambda kk: render_spiral(kk, 1.5)),
                         ('edge(0)', render_edge), ('corner(0)', render_corner)]:
            p = fn(k)
            w, mag, pts = winding_number(p, radius=1)
            # invariance: rotation adds constant (|w| same), flip flips sign (|w| same)
            wr = [abs(winding_number(rot90(p, n), 1)[0]) for n in range(1, 4)]
            wf = abs(winding_number(flip(p), 1)[0])
            print('  %-14s w=%+.3f  |w|  | rot90(|w|)=%s  flip|w|=%.3f  ringpts=%d  mean|grad|=%.2f'
                  % (name, w, ['%.3f' % v for v in wr], wf, len(pts), mag))
        print('  cos-vortex render:')
        show(render_cos_vortex(k))
    print()
    print('VERDICT: cos-vortex / spiral should give |w|~1 (topological charge); edge/corner ~0.')
    print('If |w| for vortices is far from 1 or unstable across rot/flip, WINDKEY is infeasible')
    print('at this resolution (collapses to fixed texture) -> drop, per user priority ordering.')


if __name__ == '__main__':
    main()

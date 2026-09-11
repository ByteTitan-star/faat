"""ORBIT-IRREP self-test: (1) D4 signature invariance (pure math, CPU); (2) build base b and
visualize orbit distinctness; (3) optional short proxy-refine + proxy-ASR (GPU).
Run from repo root:
  python -m faat._orbit_selftest            # math + build (+ GPU proxy-refine)
  NOGPU=1 python -m faat._orbit_selftest    # math + build only
"""
import os
import torch

from faat.orbit_trigger import (
    apply_d4, orbit_energy_signature, verify_invariance, orbit_distinct, make_base,
    build_orbit_trigger, D4_NAMES)


def math_test():
    torch.manual_seed(0)
    print('=== (1) D4 irrep-signature invariance (CPU) ===')
    for k in (5, 7):
        b = torch.rand(3, k, k)
        ref = orbit_energy_signature(b, b)
        max_rel = 0.0
        for g0 in range(8):
            sig = orbit_energy_signature(apply_d4(b, g0), b)
            for name in ref:
                rel = abs(ref[name] - sig[name]) / max(1.0, abs(ref[name]))
                max_rel = max(max_rel, rel)
        verify_invariance(b)
        print('  k=%d: max rel deviation across 8 orbit elements = %.2e  (invariance OK)'
              % (k, max_rel))
        print('    ref signature: ' + ', '.join('%s=%.3e' % (n, ref[n]) for n in ['A1', 'A2', 'B1', 'B2', 'E']))


def build_test():
    print('=== (2) build asymmetric high-contrast base b ===')
    for k in (5, 7):
        b = make_base(k=k, channels=3, seed=0)
        print('  k=%d: distinct orbit elements = %d/8 (want 8 = full asymmetry)' % (k, orbit_distinct(b)))
        print('  base pattern (channel 0):');
        for row in b[0].tolist():
            print('    ' + ' '.join('%d' % int(v) for v in row))


def gpu_test():
    if os.environ.get('NOGPU'):
        print('=== (3) GPU proxy-refine skipped (NOGPU=1) ==='); return
    from torchvision import datasets, transforms
    from cifar_resnet import ResNet18
    dev = 'cuda'
    print('=== (3) build + short proxy-refine of b (GPU, informational) ===')
    print('    NOTE: proxy-ASR is NOT a meaningful metric for small patch triggers')
    print('    (Badnets-style patches are learned by the victim, not the proxy).')
    ds = datasets.CIFAR10(root='./data', train=True, transform=transforms.ToTensor(), download=False)
    imgs = torch.stack([ds[i][0] for i in range(2000)]).to(dev)
    ck = torch.load('resource/faat/proxy/resnet18_clean_cifar10.pth', map_location=dev)
    proxy = ResNet18(num_classes=10).to(dev); proxy.load_state_dict(ck['state_dict']); proxy.eval()
    for p in proxy.parameters():
        p.requires_grad_(False)
    info = build_orbit_trigger(k=5, channels=3, loc=(21, 21), seed=0, proxy=proxy, images=imgs,
                               target=0, device=dev, opt_steps=300)
    sig = verify_invariance(info['base'].cpu())
    print('  refined-b signature: ' + ', '.join('%s=%.3e' % (n, sig[n]) for n in ['A1', 'A2', 'B1', 'B2', 'E']))


if __name__ == '__main__':
    math_test()
    build_test()
    gpu_test()

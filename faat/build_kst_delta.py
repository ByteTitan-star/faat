"""Build per-dataset KST (4th-order flat-spectrum) deltas at eps in {16,20}/255.
Reuses KST class + fit_zca from kst_sdt_validate. Delta is dataset-specific
(ZCA + 4th-order objective on that dataset's clean images). All datasets 32x32.

Usage: python faat/build_kst_delta.py --dataset cifar100
       python faat/build_kst_delta.py --dataset gtsrb  --data_dir ./data/GTSRB32 --num_classes 43
       python faat/build_kst_delta.py --dataset tiny   --data_dir ./data_tiny   --num_classes 200
"""
import argparse, os, sys
import torch
from torchvision import datasets, transforms

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from faat.kst_sdt_validate import fit_zca, KST, RES  # noqa: E402


def load_clean(dataset, data_dir, num_classes):
    tf = transforms.Compose([transforms.Resize(32), transforms.ToTensor()])
    if dataset == "cifar10":
        ds = datasets.CIFAR10(os.path.join(ROOT, "data"), train=True, transform=tf, download=False)
    elif dataset == "cifar100":
        ds = datasets.CIFAR100(os.path.join(ROOT, "data100"), train=True, transform=tf, download=False)
    else:  # gtsrb / tiny -> ImageFolder (train/)
        ds = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=tf)
    loader = torch.utils.data.DataLoader(ds, batch_size=1000, shuffle=False, num_workers=4)
    xs = []
    for x, _ in loader:
        xs.append(x)
    return torch.cat(xs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--num_classes", type=int, default=10)
    ap.add_argument("--ranks", type=int, default=4)
    args = ap.parse_args()
    device = "cuda:0"

    print(f"loading {args.dataset} clean images...", flush=True)
    x = load_clean(args.dataset, args.data_dir, args.num_classes)
    print(f"  {len(x)} images {tuple(x.shape)}", flush=True)

    mu, W = fit_zca(x)
    kst = KST(r=args.ranks, device=device)
    for eps in (16 / 255.0, 20 / 255.0):
        cache = os.path.join(RES, f"kst_delta_{args.dataset}_r{args.ranks}_eps{eps:.4f}_a0.0_ce0.pt")
        if os.path.exists(cache):
            print(f"  eps={eps:.4f}: cache exists ({cache}), skip"); continue
        delta = kst.build(x, mu, W, eps, steps=400, batch=512, device=device, cache_tag=args.dataset)
        print(f"  eps={eps:.4f}: built -> {cache} (Linf={delta.abs().max().item():.4f})", flush=True)
    print("done.")


if __name__ == "__main__":
    main()

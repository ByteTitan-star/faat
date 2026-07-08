"""Train a clean proxy of an ALTERNATIVE architecture (ResNet34/50) for the cross-arch
transfer test (path 3). The repo's proxy.py hardcodes ResNet18; this generalises it."""
import argparse
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from cifar_resnet import ResNet34, ResNet50
from faat.proxy import set_random_seed

DEV = 'cuda'
ARCH = {'resnet34': ResNet34, 'resnet50': ResNet50}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', choices=['resnet34', 'resnet50'], required=True)
    ap.add_argument('--dataset', default='cifar10')
    ap.add_argument('--data_dir', default='./data')
    ap.add_argument('--num_classes', type=int, default=10)
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--save_path', required=True)
    ap.add_argument('--seed', type=int, default=1)
    args = ap.parse_args()
    set_random_seed(args.seed)
    tf = transforms.Compose([transforms.Pad(4), transforms.RandomHorizontalFlip(),
                             transforms.RandomCrop(32), transforms.ToTensor()])
    ds = datasets.CIFAR10(root='./data', train=True, transform=tf, download=True) if args.dataset == 'cifar10' \
        else datasets.ImageFolder(root=f'{args.data_dir}/train', transform=tf)
    loader = DataLoader(ds, batch_size=128, shuffle=True, num_workers=4)
    net = ARCH[args.arch](num_classes=args.num_classes).to(DEV)
    opt = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, nesterov=True, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[30, 45], gamma=0.1)
    crit = nn.CrossEntropyLoss().to(DEV)
    for ep in range(args.epochs):
        net.train(); cor = 0
        for x, y in loader:
            x, y = x.to(DEV), y.to(DEV)
            opt.zero_grad(); loss = crit(net(x), y); loss.backward(); opt.step()
            cor += net(x).argmax(1).eq(y).sum().item()
        sched.step()
        if ep % 10 == 0 or ep == args.epochs - 1:
            print('[proxy-%s] ep %03d acc %.3f' % (args.arch, ep, cor / len(ds)))
    import os
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save({'state_dict': net.state_dict(), 'model_name': args.arch,
                'num_classes': args.num_classes, 'clean': True}, args.save_path)
    print('[proxy-%s] saved -> %s' % (args.arch, args.save_path))


if __name__ == '__main__':
    main()

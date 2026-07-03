"""GTSRB 预处理：变尺寸 .ppm -> Resize(32) PNG，按类 90/10 切 train/val，
输出标准 ImageFolder 结构 data/GTSRB32/{train,val}/<class>/*.png。
(repo 的 ResNet 是 32x32；GTSRB Test 是无类标平铺竞赛格式，故从 Train 切 val。)
Run: python -m faat.gtsrb_prep
"""
import os
import glob
import random

from PIL import Image

SRC = 'data/GTSRB/Train'
DST = 'data/GTSRB32'
SIZE = 32
VAL_FRAC = 0.1
SEED = 1


def main():
    random.seed(SEED)
    classes = sorted([d for d in os.listdir(SRC) if os.path.isdir(os.path.join(SRC, d))])
    assert len(classes) == 43, 'expected 43 classes, got %d' % len(classes)
    total = 0
    for ci, c in enumerate(classes):
        files = sorted(f for f in glob.glob(os.path.join(SRC, c, '*'))
                       if f.lower().endswith(('.ppm', '.png', '.jpg', '.jpeg')))
        random.shuffle(files)
        n_val = max(1, int(len(files) * VAL_FRAC))
        val_files = files[:n_val]
        train_files = files[n_val:]
        for split, flist in [('train', train_files), ('val', val_files)]:
            outd = os.path.join(DST, split, c)
            os.makedirs(outd, exist_ok=True)
            for f in flist:
                img = Image.open(f).convert('RGB').resize((SIZE, SIZE), Image.BILINEAR)
                img.save(os.path.join(outd, os.path.splitext(os.path.basename(f))[0] + '.png'))
        total += len(files)
        if ci % 10 == 0:
            print('[gtsrb-prep] class %d/%d  %s  train=%d val=%d' % (ci, len(classes), c, len(train_files), len(val_files)))
    print('[gtsrb-prep] DONE  %d classes, %d images -> %s' % (len(classes), total, DST))


if __name__ == '__main__':
    main()

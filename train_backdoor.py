import argparse
import numpy as np
import time
import os
import json
import torch
import re
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import heapq
import logging
from cifar_resnet import ResNet18, ResNet50, ResNet34
from utils import *
from PIL import Image

# Silence PIL's per-chunk PNG decode debug spam. GTSRB ImageFolder loads tens of
# thousands of PNGs; with the root logger at DEBUG (set below) PIL emits millions
# of "STREAM b'IHDR'/b'IDAT'" lines -> ~111MB log per run. Victim training is a
# separate subprocess, so this must be silenced here too (not only in train_faat).
for _pil_logger in ('PIL', 'PIL.PngImagePlugin', 'PIL.ImageFile'):
    logging.getLogger(_pil_logger).setLevel(logging.WARNING)

def train_step(model, criterion, optimizer, data_loader):
    model.train()
    total_correct = 0
    total_loss = 0.0
    for i, (images, labels, is_poison) in enumerate(data_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        pred = output.data.max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    loss = total_loss / len(data_loader)
    acc = float(total_correct) / len(data_loader.dataset)
    return loss, acc

def test_step(model, criterion, data_loader, target):
    model.eval()
    total_correct = 0
    total_loss = 0.0
    target_correct = 0
    target_num = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            total_loss += criterion(output, labels).item()
            pred = output.data.max(1)[1]
            total_correct += pred.eq(labels.data.view_as(pred)).sum()
            for i in range(len(pred)):
                a = pred[i].long()
                b = labels.data.view_as(pred)[i].long()
                if b == target:
                    if a == b:
                        target_correct += 1
                    target_num += 1
    loss = total_loss / len(data_loader)
    acc = float(total_correct) / len(data_loader.dataset)
    if target_num != 0:
        tar_acc = float(target_correct) / target_num
    else:
        tar_acc = 0
    return loss, acc, tar_acc

def attach_index(path, index, suffix=""):
    if re.search(suffix + "$", path):
        prefix, suffix = re.match(f"^(.*)({suffix})$", path).groups()
    else:
        prefix, suffix = path, ""
    return f"{prefix}_{index}{suffix}"

parser = argparse.ArgumentParser(description='Evaluate backdoor attack with different selection methods')
parser.add_argument('--model', default='resnet18', choices=['resnet18', 'resnet50', 'resnet34'])
parser.add_argument('--selection', default='res', choices=['random', 'loss', 'grad', 'forget', 'res', 'stealth'])
parser.add_argument('--res_sel', default='linear', choices=['max', 'exp', 'linear', 'log', 'square', 'num', 'third', 'poison'])
parser.add_argument('--batch_size', type=int, default=128, help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=300, help='number of epochs to train (default: 200)')
parser.add_argument('--learning_rate', type=float, default=0.1, help='learning rate')
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--output_dir', type=str, default='./resource/save_metric_10_res', help='directory where to save metrics, same with the one used in cal_metric.py')
parser.add_argument('--result_dir', type=str, default='save_metric_nar_log_0.00004', help='directory where to save results')
parser.add_argument('--model_dir', type=str, default='./models/', help='directory where to save results')
parser.add_argument('--y_target', type=int, default=1)
parser.add_argument('--dataset', default='cifar10', help='dataset')
parser.add_argument('--num_levels', type=str, default="36:60:12")
parser.add_argument('--poison_rate', type=float, default=0.01)
parser.add_argument('--res_rate', type=float, default=1)
parser.add_argument('--backdoor_type', default='narcissus', choices=['badnets', 'blend', 'quantize', 'narcissus', 'siba', 'faat', 'faatb', 'icit', 'rkt'])
parser.add_argument('--select_epoch', type=int, default=10, help='epoch which to calculate the stats')
parser.add_argument('--num_classes', type=int, default=10, help='num of the classes')
parser.add_argument('--blend_size', type=int, default=32, help='the size of blend image')
parser.add_argument('--data_dir', type=str, default='/home/amax/STU/DATASET/tiny-imagenet-200', help='directory where is dataset')
parser.add_argument('--type', type=str, default="0:0:0")
parser.add_argument('--faat_global_scale', type=float, default=1.0, help='FAAT: scale of delta_global (Narcissus noise); reduce for stealth')
parser.add_argument('--faat_eps', type=float, default=None, help='FAAT: override L2 budget of adaptive residual (default None = use texture rule)')
parser.add_argument('--faat_save_trigger', type=str, default=None, help='FAAT/faatb: artifact dir with global_delta.npy+adaptive_delta.npy. None=default ./resource/faat/save_trigger_{nc}_{yt}')
parser.add_argument('--icit_save_trigger', type=str, default=None, help='ICIT: artifact dir with gen.pth (trained input-conditioned generator)')
parser.add_argument('--icit_budget', type=float, default=1.5, help='ICIT: per-image L2 budget of the generator perturbation')
parser.add_argument('--strong_aug', action='store_true', help='Path-3b: add ColorJitter+RandomErasing to train_transform (survival test: does the trigger survive strong aug?)')
parser.add_argument('--rkt_save_trigger', type=str, default=None, help='RKT: artifact dir with rkt.pth (trained resampling kernel)')
parser.add_argument('--rkt_scale', type=float, default=0.7, help='RKT: resampling downscale factor s')
parser.add_argument('--rkt_ksize', type=int, default=5, help='RKT: interpolation kernel size k')
args = parser.parse_args()
use_cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda" if use_cuda else "cpu")
cudnn.benchmark = True
set_random_seed(args.seed)

if args.dataset == 'cifar10':
    if getattr(args, 'strong_aug', False):
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Pad(4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.2))])
    else:
        train_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Pad(4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32),
                transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transforms.ToTensor(), download=True)
    num_classes = 10
    test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transforms.ToTensor(), download=True)
elif args.dataset == 'cifar100':
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Pad(4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32),
        transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.CIFAR100(root='./data100', train=True, transform=transforms.ToTensor(), download=True)
    num_classes = 100
    test_dataset = datasets.CIFAR100(root='./data100', train=False, transform=transforms.ToTensor(), download=True)
else :
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Pad(4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32),
        transforms.ToTensor()])
    num_classes = args.num_classes
    train_dataset = datasets.ImageFolder(root=os.path.join(args.data_dir, 'train'), transform=transforms.ToTensor())
    test_dataset = datasets.ImageFolder(root=os.path.join(args.data_dir, 'val'), transform=transforms.ToTensor())

if args.backdoor_type == 'badnets':
    checkboards = {}
    checkboards[0] = torch.Tensor([[0,0,1],[0,1,0],[1,0,1]]).repeat((3,1,1))
    checkboards[1] = torch.Tensor([[0, 0, 0], [0, 0, 0], [0, 0, 0]]).repeat((3, 1, 1))
    checkboards[2] = torch.Tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]]).repeat((3, 1, 1))
    trigger = torch.zeros([3, 32, 32])
    if args.type != None:
        trigger_alpha = torch.zeros([3, 32, 32])
        trigger_alpha[:, 26:29, 17:20] = 1.0
    else:
        checkboard = torch.Tensor([[0,0,0],[0,0,0],[0,0,0]]).repeat((3,1,1))
        trigger[:, 26:29, 17:20] = checkboard
        trigger_alpha = torch.zeros([3, 32, 32])
        trigger_alpha[:, 26:29, 17:20] = 1.0
elif args.backdoor_type == 'blend':
    image_path = './resource/hello_kitty.jpeg'  #Replace it with the path of your JPEG image
    image = Image.open(image_path).convert('RGB')
    resized_image = image.resize((args.blend_size, args.blend_size), Image.ANTIALIAS)
    trigger_m = np.array(resized_image)
    trigger_m = torch.from_numpy(trigger_m)
    trigger_m = np.transpose(trigger_m, (2, 0, 1))
    trigger_m = trigger_m/255
    trigger_m = trigger_m.type(torch.FloatTensor)
    trigger= torch.zeros([3, 32, 32])
    trigger[:, 32 - args.blend_size:32, 32 - args.blend_size:32] = trigger_m
    trigger_alpha = torch.zeros([3, 32, 32])
    trigger_alpha[:, 32 - args.blend_size:32, 32 - args.blend_size:32] = 1.0
    if args.blend_size > 24:
        trigger_alpha *= 0.1
    elif args.blend_size > 16:
        trigger_alpha *= 0.4
    elif args.blend_size > 8:
        trigger_alpha *= 0.8
    else:
        trigger_alpha *= 1
elif args.backdoor_type == 'narcissus':
    save_path = './resource/narcissus/noise_01000.pth'
    if os.path.exists(save_path):
        temp_trigger = torch.load(save_path)
    trigger = temp_trigger.squeeze(0)

total_poison = int(len(train_dataset) * args.poison_rate)

if args.selection in ['loss', 'grad', 'forget', 'res', 'stealth']:
    stats_metric, stats_class, stats_inds = get_stats(args.selection, args.output_dir, args.select_epoch, args.seed, num_classes, args.y_target, args.res_sel, args.res_rate)
    metric_vals, metric_inds = [], []
    for i in range(len(train_dataset)):
        if stats_class[i] == args.y_target:
            metric_vals.append(stats_metric[i])
            metric_inds.append(stats_inds[i])
    if args.selection != 'stealth':
        largest_inds = heapq.nlargest(total_poison, range(len(metric_vals)), metric_vals.__getitem__)
        poison_inds = [metric_inds[i] for i in largest_inds]
else:
    shuffle = np.random.permutation(len(train_dataset))
    k = 0
    poison_inds = []
    total_poison = len(train_dataset) * args.poison_rate
    for i in shuffle:
        if args.selection == 'poison':
            if train_dataset[i][1] != args.y_target and k < total_poison:
                poison_inds.append(i)
                k += 1
        else:
            if train_dataset[i][1] == args.y_target and k < total_poison:
                poison_inds.append(i)
                k += 1
os.makedirs(args.result_dir, exist_ok=True)
logger = logging.getLogger()
if args.selection != 'stealth':
    args.poisoning_rate = len(poison_inds) *1.0/len(train_dataset)
    logger.info(str(len(poison_inds)))
logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(os.path.join(args.result_dir, 'output_{}.log'.format(args.seed))),
            logging.StreamHandler()
        ])
logger.info(args)
if args.model == 'resnet18':
    model = ResNet18(num_classes=num_classes)
elif args.model == 'resnet50':
    model = ResNet50(num_classes=num_classes)
elif args.model == 'resnet34':
    model = ResNet34(num_classes=num_classes)
#You may need to manually modify the output dimension of the last layer of the following network to adapt to the dataset.
elif args.model == 'AlexNet':
    from models.AlexNet import *
    model = AlexNet()
elif args.model == "SqueezeNet":
    from models.SqueezeNet import *
    model = SqueezeNet()
elif args.model == "VGG16":
    from models.VGG16 import *
    model = VGG16()
elif args.model == "GoogLeNet":
    from models.GoogLeNet import *
    model = GoogLeNet()
elif args.model == "DenseNet121":
    from models.DenseNet import *
    model = DenseNet121()
elif args.model == "MobileNet":
    from models.MobileNet import *
    model = MobileNet()
model = model.cuda()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
criterion = torch.nn.CrossEntropyLoss().to(device)
if args.backdoor_type == 'quantize':
    poison_train_set = Add_Clean_Label_Train_Trigger_Quantize(train_dataset, args.y_target, poison_inds, args.num_levels)
    poison_test_set = Add_Test_Trigger_Quantize(test_dataset, args.y_target, args.num_levels)
elif args.backdoor_type == 'narcissus':
    poison_train_set = Add_Clean_Label_Train_Trigger_NAR(train_dataset, trigger, args.y_target, poison_inds)
    poison_test_set = Add_Test_Trigger_NAR(test_dataset, trigger, args.y_target)
elif args.backdoor_type == 'siba':
    args.save_trigger = "./resource/save_trigger_" + str(num_classes) + "_" + str(args.y_target)
    poison_train_set = Add_Clean_Label_Train_Trigger_siba(train_dataset, args.y_target, poison_inds, args.save_trigger)
    poison_test_set = Add_Test_Trigger_siba(test_dataset, args.y_target, args.save_trigger)
elif args.backdoor_type == 'badnets':
    poison_train_set = Add_Clean_Label_Train_Trigger_badnets(train_dataset, trigger, args.y_target, trigger_alpha, poison_inds, args.type, checkboards)
    poison_test_set = Add_Test_Trigger_badnets(test_dataset, trigger, args.y_target, trigger_alpha, args.type, checkboards)
elif args.backdoor_type == 'faat':
    from faat.apply_trigger import Add_Clean_Label_Train_Trigger_faat, Add_Test_Trigger_faat
    args.save_trigger = args.faat_save_trigger or ("./resource/faat/save_trigger_" + str(num_classes) + "_" + str(args.y_target))
    poison_train_set = Add_Clean_Label_Train_Trigger_faat(train_dataset, args.y_target, poison_inds, args.save_trigger, args.faat_global_scale, args.faat_eps)
    poison_test_set = Add_Test_Trigger_faat(test_dataset, args.y_target, args.save_trigger, args.faat_global_scale)
elif args.backdoor_type == 'faatb':
    from faat.apply_trigger import Add_Clean_Label_Train_Trigger_faatb, Add_Test_Trigger_faatb
    args.save_trigger = args.faat_save_trigger or ("./resource/faat/save_trigger_" + str(num_classes) + "_" + str(args.y_target))
    poison_train_set = Add_Clean_Label_Train_Trigger_faatb(train_dataset, args.y_target, poison_inds, args.save_trigger, args.faat_global_scale)
    poison_test_set = Add_Test_Trigger_faatb(test_dataset, args.y_target, args.save_trigger, args.faat_global_scale)
elif args.backdoor_type == 'icit':
    from faat.apply_ic import Add_Clean_Label_Train_Trigger_icit, Add_Test_Trigger_icit
    args.save_trigger = args.icit_save_trigger or ("./resource/faat/icit_" + str(num_classes) + "_" + str(args.y_target))
    poison_train_set = Add_Clean_Label_Train_Trigger_icit(train_dataset, args.y_target, poison_inds, args.save_trigger, device)
    poison_test_set = Add_Test_Trigger_icit(test_dataset, args.y_target, args.save_trigger, device)
elif args.backdoor_type == 'rkt':
    from faat.apply_rkt import Add_Clean_Label_Train_Trigger_rkt, Add_Test_Trigger_rkt
    args.save_trigger = args.rkt_save_trigger or ("./resource/faat/rkt_" + str(num_classes) + "_" + str(args.y_target))
    poison_train_set = Add_Clean_Label_Train_Trigger_rkt(train_dataset, args.y_target, poison_inds, args.save_trigger, device)
    poison_test_set = Add_Test_Trigger_rkt(test_dataset, args.y_target, args.save_trigger, device)
else:
    if args.selection == 'stealth':
        poison_train_set = Add_Clean_Label_Train_Trigger_blend_stealth(train_dataset, trigger, args.y_target,
                                                               metric_inds, args.type, total_poison)
    else :
        poison_train_set = Add_Clean_Label_Train_Trigger(train_dataset, trigger, args.y_target, trigger_alpha, poison_inds, args.type)
    poison_test_set = Add_Test_Trigger(test_dataset, trigger, args.y_target, trigger_alpha, args.type)
poison_train_set = MyDataset(poison_train_set, train_transform)

train_loader = DataLoader(poison_train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
trigger_loader = DataLoader(poison_test_set, batch_size=args.batch_size, shuffle=True, num_workers=4)

if args.dataset == 'cifar10':
    model_optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, nesterov=True,
                                      weight_decay=5e-4)
    scheduler = MultiStepLR(model_optimizer, milestones=[60, 90], gamma=0.1)
else:
    model_optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, nesterov=False,
                                      weight_decay=5e-4)
    scheduler = MultiStepLR(model_optimizer, milestones=[150, 225], gamma=0.1)

os.makedirs(args.result_dir, exist_ok=True)
logger = logging.getLogger()
if args.selection != 'stealth':
    args.poisoning_rate = len(poison_inds) *1.0/len(train_dataset)
    logger.info(str(len(poison_inds)))
logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(os.path.join(args.result_dir, 'output_{}.log'.format(args.seed))),
            logging.StreamHandler()
        ])
logger.info(args)
logger.info('Epoch \t lr \t Time \t TrainLoss \t TrainACC \t PoisonLoss \t PoisonACC \t CleanLoss \t CleanACC \t TargetSum \t CleanSum')
clean_sum = 0.0
poison_sum = 0.0
for epoch in range(args.epochs):
    start = time.time()
    lr = model_optimizer.param_groups[0]['lr']
    train_loss, train_acc = train_step(model, criterion, model_optimizer, train_loader)
    cl_test_loss, cl_test_acc, cl_tar_acc= test_step(model, criterion, test_loader, args.y_target)
    po_test_loss, po_test_acc, po_tar_acc = test_step(model, criterion, trigger_loader, args.y_target)
    clean_sum = clean_sum + cl_test_acc
    poison_sum = poison_sum + po_test_acc
    po_tar_acc = 0.0
    cl_tar_acc = 0.0
    if epoch % 20 == 0:
        po_tar_acc = poison_sum / 20
        cl_tar_acc = clean_sum / 20
        clean_sum = 0.0
        poison_sum = 0.0
    scheduler.step()
    end = time.time()
    logger.info(
            '%d \t %.3f \t %.1f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f',
            epoch, lr, end - start, train_loss, train_acc, po_test_loss, po_test_acc,
            cl_test_loss, cl_test_acc, po_tar_acc, cl_tar_acc)

# ---- Phase 0 infra: persist artifacts for downstream stealth/detection/feature analysis ----
try:
    _meta = {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
             for k, v in vars(args).items()}
    _meta['num_classes'] = num_classes
    with open(os.path.join(args.result_dir, 'args.json'), 'w') as _f:
        json.dump(_meta, _f, indent=2)
except Exception as _e:
    logger.info('save args.json failed: %s' % _e)
if 'poison_inds' in dir() and args.selection != 'stealth':
    try:
        with open(os.path.join(args.result_dir, 'poison_inds.json'), 'w') as _f:
            json.dump({'poison_inds': [int(i) for i in poison_inds],
                       'y_target': int(args.y_target),
                       'poisoning_rate': float(len(poison_inds)) / len(train_dataset)},
                      _f, indent=2)
    except Exception as _e:
        logger.info('save poison_inds.json failed: %s' % _e)
try:
    torch.save({'state_dict': model.state_dict(), 'model_name': args.model,
                'num_classes': num_classes, 'y_target': int(args.y_target),
                'backdoor_type': args.backdoor_type, 'epoch': int(args.epochs - 1)},
               os.path.join(args.result_dir, 'model_last.pth'))
    logger.info('saved model_last.pth to %s' % args.result_dir)
except Exception as _e:
    logger.info('save model_last.pth failed: %s' % _e)



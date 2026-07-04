import random
import numpy as np
import os
import torch
import pickle
import re
import math
from PIL import Image
from jupyter_core.version import pattern
from cifar_resnet import UnetGenerator
from dct import *
import cv2
import torchvision.transforms as T
from scipy.ndimage import gaussian_gradient_magnitude
def set_random_seed(seed = 10):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) 
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class MyDataset(torch.utils.data.Dataset):
   
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample, label, is_poison = self.data[idx][0], self.data[idx][1], self.data[idx][2]
        if self.transform:
            sample = self.transform(sample)
        return (sample, label, is_poison)

def Add_Test_Trigger_NAR(dataset, trigger, target):
    dataset_ = list()
    for i in range(len(dataset)):
        data = dataset[i]
        img = data[0]
        label = data[1]
        #todo why
        if label == target:
            continue
        temp_img = img * 1
        for j in range(3):
            temp_img[j, :, :] = temp_img[j, :, :] + trigger[j, :, :]
            temp_img[j, :, :] = torch.clamp(temp_img[j, :, :], 0, 1)
        dataset_.append((temp_img, target))
    return dataset_
def Add_Test_Trigger_siba(dataset, target, save_trigger):
    uap = np.load('{}/uap.npy'.format(save_trigger))
    mask = np.load('{}/mask.npy'.format(save_trigger))
    uap = torch.from_numpy(uap)
    mask = torch.from_numpy(mask)
    mask = mask.detach()
    uap = uap.detach()
    dataset_ = list()
    for i in range(len(dataset)):
        data = dataset[i]
        img = data[0]
        label = data[1]
        #todo why
        if label == target:
            continue
        temp_img = img + uap * mask
        temp_img = torch.clamp(temp_img, 0, 1)
        dataset_.append((temp_img, target))
    return dataset_
def Add_Test_Trigger_ground(dataset, target, datasetname):
    upgd_path = "/home/boot/STU/workspaces/wzx/BLearnDefense/triggers/ground/upgd-" + datasetname + "-ResNet18-Linf-eps8.0"
    trigger = torch.load(os.path.join(upgd_path, 'upgd_'+str(target)+'.pth'), map_location='cpu')
    dataset_ = list()
    for i in range(len(dataset)):
        data = dataset[i]
        img = data[0]
        label = data[1]
        #todo why
        if label == target:
            continue
        temp_img = img + trigger
        temp_img = torch.clamp(temp_img, 0, 1)
        dataset_.append((temp_img, target))
    return dataset_
def low_freq(x):
    image_size = 32
    ratio = 0.65
    mask = torch.zeros_like(x)
    mask[:, :, : int(image_size * ratio), : int(image_size * ratio)] = 1
    x_dct = dct_2d((x + 1) / 2 * 255)
    x_dct *= mask
    x_idct = (idct_2d(x_dct) / 255 * 2) - 1
    return x_idct
def Add_Clean_Label_Train_Trigger_combat(dataset, target, class_order):
    netG = None
    use_cuda = True if torch.cuda.is_available() else False
    device = torch.device("cuda" if use_cuda else "cpu")
    netG = UnetGenerator().to(device)
    gauss_smooth = T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))
    load_path = "/home/boot/STU/workspaces/wzx/Samples_select/cifar10_combat_clean.pth.tar"
    if not os.path.exists(load_path):
        print("Error: {} not found".format(load_path))
        #exit()
    else:
        state_dict = torch.load(load_path)
        netG.load_state_dict(state_dict["netG"])
        netG.eval()
    dataset_ = list()
    for i in range(len(dataset)):
        data = dataset[i]
        img = data[0]
        label = data[1]
        if i in class_order:
            img =img.to('cuda')
            noise_bd = netG(img.unsqueeze(0))
            if img.shape[0] != 0:
                noise_bd = low_freq(noise_bd)
            inputs_bd = torch.clamp(img + noise_bd * 0.08, -1, 1)
            if inputs_bd.shape[0] != 0:
                inputs_bd = gauss_smooth(inputs_bd)
            inputs_bd = inputs_bd.squeeze(0)
            for j in range(3):
                inputs_bd[j, :, :] = torch.clamp(inputs_bd[j, :, :], 0, 1)
            dataset_.append((inputs_bd, target, 1))
        else:
            dataset_.append((img, data[1], 0))           
    return dataset_
def Add_Test_Trigger_combat(dataset, target):
    netG = None
    use_cuda = True if torch.cuda.is_available() else False
    device = torch.device("cuda" if use_cuda else "cpu")
    netG = UnetGenerator().to(device)
    gauss_smooth = T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))
    load_path = "./resource/cifar10_combat_clean.pth.tar"
    if not os.path.exists(load_path):
        print("Error: {} not found".format(load_path))
        #exit()
    else:
        state_dict = torch.load(load_path)
        netG.load_state_dict(state_dict["netG"])
        netG.eval()
    dataset_ = list()
    for i in range(len(dataset)):
        data = dataset[i]
        img = data[0]
        label = data[1]
        if label == target:
            continue
        img =img.to('cuda')
        noise_bd = netG(img.unsqueeze(0))
        if img.shape[0] != 0:
            noise_bd = low_freq(noise_bd)
        inputs_bd = torch.clamp(img + noise_bd * 0.08, -1, 1)
        if inputs_bd.shape[0] != 0:
            inputs_bd = gauss_smooth(inputs_bd)
        inputs_bd = inputs_bd.squeeze(0)
        for j in range(3):
            inputs_bd[j, :, :] = torch.clamp(inputs_bd[j, :, :], 0, 1)
        dataset_.append((inputs_bd, target))
    return dataset_
def Add_Test_Trigger(dataset, trigger, target, alpha, type):
    dataset_ = list()
    if type == None:
        for i in range(len(dataset)):
            data = dataset[i]
            img = data[0]
            label = data[1]
            #todo why
            if label == target:
                continue
            img = (1-alpha)*img + alpha*trigger
            img = torch.clamp(img, 0, 1)
            dataset_.append((img, target))
    else:
        pattern = re.compile(r'(\d+):(\d+):(\d+)')
        match = pattern.match(type)
        if not match:
            raise ValueError('num_levels is not valid')
        t_R, t_G, t_B = map(int, match.groups())
        trigger_alpha = torch.zeros([3, 32, 32])
        trigger_alpha[:, 0:32, 0:32] = 1.0
        checkboard = [t_R, t_G, t_B]
        checkboard = [2, 2, 2]  # [复现补丁 2026-07-01] Blended-C 透明度改为 0.2:0.2:0.2（论文 Table1 默认）；原代码硬编码 [2,1,3]=0.2:0.1:0.3（属 Component-C 优化版/Table4）
        for i in range(len(dataset)):
            data = dataset[i]
            img = data[0]
            label = data[1]
            #todo why
            if label == target:
                continue
            temp_img = img*1
            for j in range(3):
                alpha = checkboard[j]*0.1
                temp_img[j,:,:] = (1-alpha*trigger_alpha[j,:,:])*temp_img[j,:,:] + alpha*trigger_alpha[j,:,:]*trigger[j,:,:]
                temp_img[j,:,:] = torch.clamp(temp_img[j,:,:], 0, 1)
            dataset_.append((temp_img, target))
    return dataset_
def compute_gmsd(img1_path, img2_path, sigma=1.4):
    """
    计算两张图像之间的 GMSD 值。

    参数:
    img1, img2: 输入的灰度图像，形状为 (H, W)。
    sigma: 高斯核的标准差，用于计算梯度幅值。

    返回:
    gmsd_value: 计算得到的 GMSD 值。
    """
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    if img1.shape != img2.shape:
        raise ValueError("两张图片的尺寸不一致，请调整为相同大小。")

    # 计算梯度幅值
    grad_mag1 = gaussian_gradient_magnitude(img1.astype(np.float32), sigma=sigma)
    grad_mag2 = gaussian_gradient_magnitude(img2.astype(np.float32), sigma=sigma)

    # 计算梯度幅值的相似性
    similarity_map = 2 * grad_mag1 * grad_mag2 / (grad_mag1 ** 2 + grad_mag2 ** 2 + 1e-8)

    # 计算相似性偏差
    gmsd_value = np.std(similarity_map)
    return gmsd_value
    
def Add_Clean_Label_Train_Trigger_blend_stealth(dataset, trigger, target, class_order, type, num):
    dataset_ = list()
    pattern = re.compile(r'(\d+):(\d+):(\d+)')
    match = pattern.match(type)
    if not match:
        raise ValueError('num_levels is not valid')
    t_R, t_G, t_B = map(int, match.groups())
    trigger_alpha = torch.zeros([3, 32, 32])
    trigger_alpha[:, 0:32, 0:32] = 1.0
    checkboard = [t_R, t_G, t_B]
    u = t_R + t_G + t_B + t_R*t_G*t_B
    u = u + random.randint(100*u,100*u+100)
    temp = list()
    m = 0
    test = './testblend' + str(u) + '.JPEG'
    testw = './testwblend' + str(u) + '.JPEG'
    for i in range(len(dataset)):
        data = dataset[i]
        img = data[0]
        label = data[1]
        if i in class_order:
            temp_img = img * 1
            for j in range(3):
                alpha = checkboard[j]*0.1
                temp_img[j,:,:] = (1-alpha*trigger_alpha[j,:,:])*temp_img[j,:,:] + alpha*trigger_alpha[j,:,:]*trigger[j,:,:]
                temp_img[j,:,:] = torch.clamp(temp_img[j,:,:], 0, 1)
            image = (img * 255).numpy().astype(np.uint8)
            image_np = np.moveaxis(image, 0, -1)
            cv2.imwrite(test, image_np)
            imaget = (temp_img * 255).numpy().astype(np.uint8)
            image_npt = np.moveaxis(imaget, 0, -1)
            cv2.imwrite(testw, image_npt)
            temp.append((img, label, compute_gmsd(test, testw), m))
            m = m + 1
        else:
            dataset_.append((img, data[1], 0))
    sorted_tuples = sorted(temp, key=lambda x: x[2])
    nsmallest = [t[3] for t in sorted_tuples[:num]]
    for i in range(len(temp)):
        if i in nsmallest:
            img = temp[i][0]
            temp_img = img * 1
            for j in range(3):
                alpha = checkboard[j] * 0.1
                temp_img[j, :, :] = (1 - alpha * trigger_alpha[j, :, :]) * temp_img[j, :, :] + alpha * trigger_alpha[j,
                                                                                                       :, :] * trigger[
                                                                                                               j, :, :]
                temp_img[j, :, :] = torch.clamp(temp_img[j, :, :], 0, 1)
            dataset_.append((temp_img, temp[i][1], 1))
        else:
            dataset_.append((temp[i][0], temp[i][1], 0))
    return dataset_
def Add_Clean_Label_Train_Trigger(dataset, trigger, target, alpha, class_order, type):
    dataset_ = list()
    if type == None:
        for i in range(len(dataset)):
            data = dataset[i]
            img = data[0]
            label = data[1]
            if i in class_order:
                img = (1-alpha)*img + alpha*trigger
                img = torch.clamp(img, 0, 1)
                dataset_.append((img, target, 1))
            else:
                dataset_.append((img, data[1], 0))
    else:
        pattern = re.compile(r'(\d+):(\d+):(\d+)')
        match = pattern.match(type)
        if not match:
            raise ValueError('num_levels is not valid')
        t_R, t_G, t_B = map(int, match.groups())
        trigger_alpha = torch.zeros([3, 32, 32])
        trigger_alpha[:, 0:32, 0:32] = 1.0
        checkboard = [t_R, t_G, t_B]
        checkboard = [2, 2, 2]  # [复现补丁 2026-07-01] Blended-C 透明度改为 0.2:0.2:0.2（论文 Table1 默认）；原代码硬编码 [2,1,3]=0.2:0.1:0.3（属 Component-C 优化版/Table4）
        for i in range(len(dataset)):
            data = dataset[i]
            img = data[0]
            label = data[1]
            if i in class_order:
                temp_img = img*1
                for j in range(3):
                    alpha = checkboard[j]*0.1
                    temp_img[j,:,:] = (1-alpha*trigger_alpha[j,:,:])*temp_img[j,:,:] + alpha*trigger_alpha[j,:,:]*trigger[j,:,:]
                    temp_img[j,:,:] = torch.clamp(temp_img[j,:,:], 0, 1)
                dataset_.append((temp_img, target, 1))
            else:
                dataset_.append((img, data[1], 0))          
    return dataset_
def Add_Test_Trigger_badnets(dataset, trigger, target, alpha, type, checkboards):
    dataset_ = list()
    if type == None:
        for i in range(len(dataset)):
            data = dataset[i]
            img = data[0]
            label = data[1]
            #todo why
            if label == target:
                continue
            img = (1-alpha)*img + alpha*trigger
            img = torch.clamp(img, 0, 1)
            dataset_.append((img, target))
    else:
        trigger = torch.zeros([3, 32, 32])
        dataset_ = list()
        pattern = re.compile(r'(\d+):(\d+):(\d+)')
        match = pattern.match(type)
        temp = list()
        m = 0
        if not match:
            raise ValueError('num_levels is not valid')
        t_R, t_G, t_B = map(int, match.groups())
        checkboard = [t_R, t_G, t_B]
        checkboard = [0, 0, 0]
        for i in range(len(dataset)):
            data = dataset[i]
            img = data[0]
            label = data[1]
            #todo why
            if label == target:
                continue
            temp_img = img*1
            for j in range(3):
                if checkboard[j] < 3:
                    trigger[:, 26:29, 17:20] = checkboards[checkboard[j]]
                    temp_img[j,:,:] = (1-alpha[j,:,:])*temp_img[j,:,:] + alpha[j,:,:]*trigger[j,:,:]
                    temp_img[j,:,:] = torch.clamp(temp_img[j,:,:], 0, 1)
            dataset_.append((temp_img, target)) 
    return dataset_
def Add_Clean_Label_Train_Trigger_badnets(dataset, trigger, target, alpha, class_order, type, checkboards):
    dataset_ = list()
    if type == None:
        for i in range(len(dataset)):
            data = dataset[i]
            img = data[0]
            label = data[1]
            if i in class_order:
                img = (1-alpha)*img + alpha*trigger
                img = torch.clamp(img, 0, 1)
                dataset_.append((img, target, 1))
            else:
                dataset_.append((img, data[1], 0))
    else:
        dataset_ = list()
        pattern = re.compile(r'(\d+):(\d+):(\d+)')
        match = pattern.match(type)
        temp = list()
        m = 0
        if not match:
            raise ValueError('num_levels is not valid')
        t_R, t_G, t_B = map(int, match.groups())
        checkboard = [t_R, t_G, t_B]
        checkboard = [0, 0, 0] 
        for i in range(len(dataset)):
            data = dataset[i]
            img = data[0]
            label = data[1]
            if i in class_order:
                temp_img = img*1
                for j in range(3):
                    if checkboard[j] < 3:
                        trigger[:, 26:29, 17:20] = checkboards[checkboard[j]]
                        temp_img[j,:,:] = (1-alpha[j,:,:])*temp_img[j,:,:] + alpha[j,:,:]*trigger[j,:,:]
                        temp_img[j,:,:] = torch.clamp(temp_img[j,:,:], 0, 1)
                dataset_.append((temp_img, target, 1))
            else:
                dataset_.append((img, data[1], 0))          
    return dataset_
def Add_Clean_Label_Train_Trigger_Quantize(dataset, target, class_order, num_levels):
    dataset_ = list()
    pattern = re.compile(r'(\d+):(\d+):(\d+)')
    match = pattern.match(num_levels)
    if not match:
        raise ValueError('num_levels is not valid')
    num_R, num_G, num_B = map(int, match.groups())
    step_B = 255 // (num_B - 1)
    step_G = 255 // (num_G - 1)
    step_R = 255 // (num_R - 1)
    j = 0
    for i in range(len(dataset)):
        data = dataset[i]
        img = data[0]
        label = data[1]
        if i in class_order:
            img[0, :, :] = (((img[0, :, :] * 255) // step_R + 1) * step_R) / 255
            img[1, :, :] = (((img[1, :, :] * 255) // step_G + 1) * step_G) / 255
            img[2, :, :] = (((img[2, :, :] * 255) // step_B + 1) * step_B) / 255
            img = torch.clamp(img, 0, 1)
            dataset_.append((img, target, 1))
        else:
            dataset_.append((img, data[1], 0))
    return dataset_

def Add_Clean_Label_Train_Trigger_NAR(dataset, trigger, target, class_order):
    dataset_ = list()
    checkboard = [2, 1, 3]
    for i in range(len(dataset)):
        data = dataset[i]
        img = data[0]
        label = data[1]
        if i in class_order:
            temp_img = img * 1
            for j in range(3):
                temp_img[j, :, :] = temp_img[j, :, :] + checkboard[j]*trigger[j, :, :]
                temp_img[j, :, :] = torch.clamp(temp_img[j, :, :], 0, 1)
            dataset_.append((temp_img, target, 1))
        else:
            dataset_.append((img, data[1], 0))           
    return dataset_
def Add_Clean_Label_Train_Trigger_siba(dataset, target, class_order, save_trigger):
    uap = np.load('{}/uap.npy'.format(save_trigger))
    mask = np.load('{}/mask.npy'.format(save_trigger))
    uap = torch.from_numpy(uap)
    mask = torch.from_numpy(mask)
    mask = mask.detach()
    uap = uap.detach()
    dataset_ = list()
    for i in range(len(dataset)):
        data = dataset[i]
        img = data[0]
        label = data[1]
        if i in class_order:
            temp_img = img + uap * mask
            temp_img = torch.clamp(temp_img, 0, 1)
            dataset_.append((temp_img, target, 1))
        else:
            dataset_.append((img, data[1], 0))           
    return dataset_
def Add_Test_Trigger_Quantize(dataset, target, num_levels):
    dataset_ = list()
    pattern = re.compile(r'(\d+):(\d+):(\d+)')
    match = pattern.match(num_levels)
    if not match:
        raise ValueError('num_levels is not valid')
    num_R, num_G, num_B = map(int, match.groups())
    step_B = 255 // (num_B - 1)
    step_G = 255 // (num_G - 1)
    step_R = 255 // (num_R - 1)
    for i in range(len(dataset)):
        data = dataset[i]
        img = data[0]
        label = data[1]
        if label == target:
            continue
        img[0, :, :] = (((img[0, :, :] * 255) // step_R + 1) * step_R) / 255
        img[1, :, :] = (((img[1, :, :] * 255) // step_G + 1) * step_G) / 255
        img[2, :, :] = (((img[2, :, :] * 255) // step_B + 1) * step_B) / 255
        img = torch.clamp(img, 0, 1)
        dataset_.append((img, target))
    return dataset_

def get_stats(selection, output_dir, epoch, seed, num_classes, target, res_sel, rate=1.0):
    if selection == 'loss':
        fname = os.path.join(output_dir, 'resnet_loss_grad_epoch_{}_seed_{}.pkl'.format(epoch, seed))
        with open(fname, "rb") as fin:
            loaded = pickle.load(fin)    
        stats_metric = loaded['loss']
        stats_class = loaded['class']
        stats_order = np.arange(len(stats_metric))
    elif selection == 'stealth':
        fname = os.path.join(output_dir, 'resnet_loss_grad_epoch_{}_seed_{}.pkl'.format(epoch, seed))
        with open(fname, "rb") as fin:
            loaded = pickle.load(fin)
        stats_metric = loaded['grad_norm']
        stats_class = loaded['class']
        stats_order = np.arange(len(stats_metric))
    elif selection == 'grad':
        fname = os.path.join(output_dir, 'resnet_loss_grad_epoch_{}_seed_{}.pkl'.format(epoch, seed))
        with open(fname, "rb") as fin:
            loaded = pickle.load(fin)    
        stats_metric = loaded['grad_norm']
        stats_class = loaded['class']
        stats_order = np.arange(len(stats_metric))
    elif selection == 'forget':
        fname = os.path.join(output_dir, 'stats_forget_seed_{}.pkl'.format(seed))
        with open(fname, 'rb') as fin:
            loaded = pickle.load(fin)
        stats_metric = loaded['forget']
        stats_class = loaded['class']
        stats_order = loaded['original_index']
    elif selection == 'res':
        fname = os.path.join(output_dir, 'stats_forget_seed_{}.pkl'.format(seed))
        with open(fname, 'rb') as fin:
            loaded = pickle.load(fin)
        cls={}
        res = loaded['res']
        sum = 0.0
        stats_class = loaded['class']
        stats_order = loaded['original_index']
        for ind in range(len(stats_order)):
            index = stats_order[ind]
            for i in range(num_classes):
                cls_res = cls.get(i, 0)
                res_index = res.get(index, {})
                value_res = res_index.get(i, 0)
                cls[i] = cls_res + value_res
                sum += value_res
        if res_sel == "linear":
            for i in range(num_classes):
                cls_res = cls.get(i, 0)
                cls[i] = 1 - rate*(cls_res * 1.0 / sum)
        elif res_sel == "max" or res_sel == "num":
            sum = 0
            for i in range(num_classes):
                cls_res = cls.get(i, 0)
                sum = sum + 1.0 * math.exp(-cls_res)
            for i in range(num_classes):
                cls_res = cls.get(i, 0)
                cls[i] = 1.0 * math.exp(-cls_res) / sum
        elif res_sel == "exp":
            # 数值稳定 softmax: exp(-cls_res) 在 cls_res~1e4 时下溢为0导致除零;
            # 减去 cls 最小值(等价于 softmax 的平移不变性),数学结果不变。
            cmin = min(cls.get(i, 0) for i in range(num_classes))
            sum = 0
            for i in range(num_classes):
                sum = sum + 1.0 * math.exp(-(cls.get(i, 0) - cmin))
            for i in range(num_classes):
                cls_res = cls.get(i, 0)
                cls[i] = 1.0 * math.exp(-(cls_res - cmin)) / sum
        elif res_sel == "log":
            sum = 0
            for i in range(num_classes):
                cls_res = cls.get(i, 0)
                sum = sum + 1.0 * math.log(1 + cls_res)
            for i in range(num_classes):
                cls_res = cls.get(i, 0)
                cls[i] = 1 - rate*1.0 * math.log(1 + cls_res) / sum
        elif res_sel == "square":
            sum = 0
            for i in range(num_classes):
                cls_res = cls.get(i, 0)
                sum = sum +  cls_res*cls_res
            for i in range(num_classes):
                cls_res = cls.get(i, 0)
                cls[i] = 1 - rate*1.0 * cls_res*cls_res / sum
        elif res_sel == "third":
            sum = 0
            for i in range(num_classes):
                cls_res = cls.get(i, 0)
                sum = sum +  cls_res*cls_res*cls_res
            for i in range(num_classes):
                cls_res = cls.get(i, 0)
                cls[i] = 1 - rate*1.0 * cls_res*cls_res*cls_res / sum
        stats_metric = []
        if res_sel == "num":
            for index in stats_order:
                index_sum = 0
                for i in range(num_classes):
                    cls_res = cls.get(i, 0)
                    res_index = res.get(index, {})
                    value_res = res_index.get(i, 0)
                    if value_res > 0:
                        index_sum += cls_res
                stats_metric.append(index_sum)
        else:
            for index in stats_order:
                index_sum = 0
                for i in range(num_classes):
                    cls_res = cls.get(i, 0)
                    res_index = res.get(index, {})
                    value_res = res_index.get(i, 0)
                    if res_sel == "max":
                        index_sum = max(index_sum, cls_res*value_res)
                    else:
                        index_sum = index_sum + cls_res * value_res
                stats_metric.append(index_sum)
    else:
        raise ValueError('Unknown selection {}'.format(selection))
    return stats_metric, stats_class, stats_order

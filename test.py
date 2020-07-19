from __future__ import print_function
import argparse
import torch
from model import DLN
import torchvision.transforms as transforms
import numpy as np
from os.path import join
import time
import math
from lib.dataset import is_image_file
from PIL import Image
from os import listdir

import os

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=1, help="super resolution upscale factor")
parser.add_argument('--testBatchSize', type=int, default=32, help='testing batch size')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--chop_forward', type=bool, default=True)
parser.add_argument('--patch_size', type=int, default=256, help='0 to use original frame size')
parser.add_argument('--stride', type=int, default=16, help='0 to use original patch size')
parser.add_argument('--threads', type=int, default=1, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=1, type=int, help='number of gpu')
parser.add_argument('--image_dataset', type=str, default='test_img')
parser.add_argument('--model_type', type=str, default='DLN')
parser.add_argument('--output', default='./output/', help='Location to save checkpoint models')
parser.add_argument('--modelfile', default='models/DLN_pretrained.pth', help='sr pretrained base model')
parser.add_argument('--image_based', type=bool, default=True, help='use image or video based ULN')
parser.add_argument('--chop', type=bool, default=False)

opt = parser.parse_args()

gpus_list = range(opt.gpus)
print(opt)

cuda = opt.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Building model ', opt.model_type)

model = DLN()
model = torch.nn.DataParallel(model, device_ids=gpus_list)
model.load_state_dict(torch.load(
    opt.modelfile,
    map_location=lambda storage, loc: storage))
if cuda:
    model = model.cuda(gpus_list[0])


def eval():
    model.eval()
    LL_filename = os.path.join(opt.image_dataset)
    est_filename = os.path.join(opt.output)
    try:
        os.stat(est_filename)
    except:
        os.mkdir(est_filename)
    LL_image = [join(LL_filename, x) for x in sorted(listdir(LL_filename)) if is_image_file(x)]
    print(LL_filename)
    Est_img = [join(est_filename, x) for x in sorted(listdir(LL_filename)) if is_image_file(x)]
    print(Est_img)
    trans = transforms.ToTensor()
    channel_swap = (1, 2, 0)
    time_ave = 0
    for i in range(LL_image.__len__()):
        with torch.no_grad():
            LL_in = Image.open(LL_image[i]).convert('RGB')
            LL = trans(LL_in)
            LL = LL.unsqueeze(0)
            LL = LL.cuda()
            t0 = time.time()
            prediction = model(LL)
            t1 = time.time()
            time_ave += (t1 - t0)
            prediction = prediction.data[0].cpu().numpy().transpose(channel_swap)

            prediction = prediction * 255.0
            prediction = prediction.clip(0, 255)
            Image.fromarray(np.uint8(prediction)).save(Est_img[i])
            print("===> Processing Image: %04d /%04d in %.4f s." % (i, LL_image.__len__(), (t1 - t0)))

    print("===> Processing Time: %.4f ms." % (time_ave / LL_image.__len__() * 1000))


def modcrop(img, modulo):
    (ih, iw) = img.size
    ih = ih - (ih % modulo)
    iw = iw - (iw % modulo)
    img = img.crop((0, 0, ih, iw))

    return img


def rgb2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        float32, [0, 255]
        float32, [0, 255]
    '''
    img.astype(np.float32)
    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                              [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    rlt = rlt.round()

    return rlt


def PSNR(pred, gt, shave_border):
    pred = pred[shave_border:-shave_border, shave_border:-shave_border]
    gt = gt[shave_border:-shave_border, shave_border:-shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)


transform = transforms.Compose([
    transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
]
)

##Eval Start!!!!
eval()

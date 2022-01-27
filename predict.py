# PyTorch lib
import argparse
import os

import cv2
# Tools lib
import numpy as np
import torch
from torch.autograd import Variable

# Models lib
# Metrics lib
from metrics import calc_psnr, calc_ssim
from models import *

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default='test', type=str)
    parser.add_argument("--input_dir", default='./data/test_a/data/', type=str)
    parser.add_argument("--output_dir", default='./data/test_a/derain/', type=str)
    parser.add_argument("--gt_dir", default='./data/test_a/gt/', type=str)
    parser.add_argument("--mask_dir", default='./data/test_a/mask/', type=str)
    args = parser.parse_args()
    return args


def align_to_four(img):
    # print ('before alignment, row = %d, col = %d'%(img.shape[0], img.shape[1]))
    # align to four
    a_row = int(img.shape[0] / 4) * 4
    a_col = int(img.shape[1] / 4) * 4
    img = img[0:a_row, 0:a_col]
    # print ('after alignment, row = %d, col = %d'%(img.shape[0], img.shape[1]))
    return img


def predict(image, mask):
    image = np.array(image, dtype='float32') / 255.
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :, :, :]
    image = torch.from_numpy(image)
    image = Variable(image).to(device)
    mask = np.array(mask, dtype='float32') / 255.
    mask = mask[np.newaxis, np.newaxis, :, :]
    mask = torch.from_numpy(mask)
    mask = Variable(mask).to(device)

    out = model(image, mask)[-1]

    out = out.cpu().data
    out = out.numpy()
    out = out.transpose((0, 2, 3, 1))
    out = out[0, :, :, :] * 255.

    return out


if __name__ == '__main__':
    args = get_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Generator1().to(device)
    checkpoint = torch.load('/content/drive/MyDrive/code/DeRaindrop-master/trains_out/2_generator.pth.tar')
    model.load_state_dict(checkpoint['model_state_dict'])
    if args.mode == 'demo':
        input_list = sorted(os.listdir(args.input_dir))
        num = len(input_list)
        for i in range(num):
            print('Processing image: %s' % (input_list[i]))
            img = cv2.imread(args.input_dir + input_list[i])
            img = align_to_four(img)
            result = predict(img)
            img_name = input_list[i].split('.')[0]
            cv2.imwrite(args.output_dir + img_name + '.jpg', result)

    elif args.mode == 'test':
        input_list = sorted(os.listdir(args.input_dir))
        gt_list = sorted(os.listdir(args.gt_dir))
        mask_list = sorted(os.listdir(args.mask_dir))
        num = len(input_list)
        cumulative_psnr = 0
        cumulative_ssim = 0
        for i in range(num):
            print('Processing image: %s' % (input_list[i]))
            img = cv2.imread(args.input_dir + input_list[i])
            gt = cv2.imread(args.gt_dir + gt_list[i])
            mask = cv2.imread(args.mask_dir + mask_list[i], cv2.IMREAD_GRAYSCALE)
            img = align_to_four(img)
            gt = align_to_four(gt)
            mask = align_to_four(mask)
            dsize = (360, 240)
            img = cv2.resize(img, dsize)
            gt = cv2.resize(gt, dsize)
            mask = cv2.resize(mask, dsize)
            result = predict(img, mask)
            result = np.array(result, dtype='uint8')
            cur_psnr = calc_psnr(result, gt)
            cur_ssim = calc_ssim(result, gt)
            print('PSNR is %.4f and SSIM is %.4f' % (cur_psnr, cur_ssim))
            cumulative_psnr += cur_psnr
            cumulative_ssim += cur_ssim
        print('In testing dataset, PSNR is %.4f and SSIM is %.4f' % (cumulative_psnr / num, cumulative_ssim / num))

    else:
        print('Mode Invalid!')

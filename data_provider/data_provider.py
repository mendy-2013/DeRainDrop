# cited from:
#             @Author  : Luo Yao
#             @Site    : http://icode.baidu.com/repos/baidu/personal-code/Luoyao
#             @File    : data_provider.py
import os.path as ops

import numpy as np
import cv2

from config import global_config

CFG = global_config.cfg


class DataSet(object):

    def __init__(self, dataset_info_file):
        self._gt_img_list, self._gt_label_list, self._gt_mask_list= self._init_dataset(dataset_info_file)
        self._random_dataset()
        self._next_batch_loop_count = 0

    def _init_dataset(self, dataset_info_file):

        gt_img_list = []
        gt_label_list = []
        gt_mask_list = []

        assert ops.exists(dataset_info_file), '{:s}　不存在'.format(dataset_info_file)

        with open(dataset_info_file, 'r') as file:
            for _info in file:
                info_tmp = _info.strip(' ').split()

                gt_img_list.append(info_tmp[0])
                gt_label_list.append(info_tmp[1])
                gt_mask_list.append(info_tmp[2])
                print(gt_img_list[-1], gt_label_list[-1], gt_mask_list[-1])
        # print(gt_img_list, gt_label_list)
        return gt_img_list, gt_label_list, gt_mask_list

    def _random_dataset(self):

        assert len(self._gt_img_list) == len(self._gt_label_list)

        random_idx = np.random.permutation(len(self._gt_img_list))
        new_gt_img_list = []
        new_gt_label_list = []
        new_gt_mask_list = []

        for index in random_idx:
            new_gt_img_list.append(self._gt_img_list[index])
            new_gt_label_list.append(self._gt_label_list[index])
            new_gt_mask_list.append(self._gt_mask_list[index])

        self._gt_img_list = new_gt_img_list
        self._gt_label_list = new_gt_label_list
        self._gt_mask_list = new_gt_mask_list

    def next_batch(self, batch_size):
        assert len(self._gt_label_list) == len(self._gt_img_list)

        idx_start = batch_size * self._next_batch_loop_count
        idx_end = batch_size * self._next_batch_loop_count + batch_size

        if idx_end > len(self._gt_label_list):
            self._random_dataset()
            self._next_batch_loop_count = 0
            return self.next_batch(batch_size)
        else:
            gt_img_list = self._gt_img_list[idx_start:idx_end]
            gt_label_list = self._gt_label_list[idx_start:idx_end]
            gt_mask_list = self._gt_mask_list[idx_start:idx_end]

            gt_imgs = []
            gt_labels = []
            mask_labels = []

            for index, gt_img_path in enumerate(gt_img_list):
                gt_image = cv2.imread(gt_img_path, cv2.IMREAD_COLOR)
                label_image = cv2.imread(gt_label_list[index], cv2.IMREAD_COLOR)
                mask_image = cv2.imread(gt_mask_list[index], cv2.IMREAD_GRAYSCALE)

                gt_image = cv2.resize(gt_image, (CFG.TRAIN.IMG_WIDTH, CFG.TRAIN.IMG_HEIGHT))
                label_image = cv2.resize(label_image, (CFG.TRAIN.IMG_WIDTH, CFG.TRAIN.IMG_HEIGHT))
                mask_image = cv2.resize(mask_image, (CFG.TRAIN.IMG_WIDTH, CFG.TRAIN.IMG_HEIGHT))
                print(mask_image.shape)

                # diff_image = np.abs(np.array(cv2.cvtColor(gt_image, cv2.COLOR_BGR2GRAY), np.float32) -
                #                     np.array(cv2.cvtColor(label_image, cv2.COLOR_BGR2GRAY), np.float32))
                #
                # mask_image = np.zeros(diff_image.shape, np.float32)
                #
                # mask_image[np.where(diff_image >= 30)] = 1

                gt_image = np.divide(gt_image, 127.5) - 1
                label_image = np.divide(label_image, 127.5) - 1
                mask_image = np.divide(mask_image, 127.5) - 1

                gt_imgs.append(gt_image)
                gt_labels.append(label_image)
                mask_labels.append(mask_image)

            self._next_batch_loop_count += 1
            return gt_imgs, gt_labels, mask_labels

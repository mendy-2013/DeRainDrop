# cited from:
#             @Author  : Luo Yao
#             @Site    : http://icode.baidu.com/repos/baidu/personal-code/Luoyao
#             @File    : test_model.py

import os.path as ops
import argparse

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr

from attentive_gan_model import derain_drop_net
from config import global_config
from skimage import measure

CFG = global_config.cfg


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str,
                        default='./data/',
                        help='The input image path')
    parser.add_argument('--weights_path', type=str,
                        default='./model/derain_gan_tensorflow99500/derain_gan_2021-11-17-09-07-39.ckpt-99500',
                        help='The model weights path')
    parser.add_argument('--label_path', type=str,
                        default=None,
                        help='The label image path')

    return parser.parse_args()


def minmax_scale(input_arr):
    min_val = np.min(input_arr)
    max_val = np.max(input_arr)

    output_arr = (input_arr - min_val) * 255.0 / (max_val - min_val)

    return output_arr


def test_model(image_path, weights_path, label_path):
    assert ops.exists(image_path)

    with tf.device('/gpu:0'):
        input_tensor = tf.placeholder(dtype=tf.float32,
                                      shape=[CFG.TEST.BATCH_SIZE, CFG.TEST.IMG_HEIGHT, CFG.TEST.IMG_WIDTH, 3],
                                      name='input_tensor'
                                      )

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (CFG.TEST.IMG_WIDTH, CFG.TEST.IMG_HEIGHT))
    image_vis = image
    image = np.divide(image, 127.5) - 1

    label_image_vis = None
    if label_path is not None:
        label_image = cv2.imread(label_path, cv2.IMREAD_COLOR)
        label_image_vis = cv2.resize(
            label_image, (CFG.TEST.IMG_WIDTH, CFG.TEST.IMG_HEIGHT), interpolation=cv2.INTER_LINEAR
        )

    phase = tf.constant('test', tf.string)

    with tf.device('/gpu:0'):
        net = derain_drop_net.DeRainNet(phase=phase)
        output, attention_maps = net.build(input_tensor=input_tensor, name='derain_net_loss')

    # Set sess configuration
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TRAIN.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    sess = tf.Session(config=sess_config)

    saver = tf.train.Saver()

    with tf.device('/gpu:0'):
        with sess.as_default():
            saver.restore(sess=sess, save_path=weights_path)

            output_image, atte_maps = sess.run(
                [output, attention_maps],
                feed_dict={input_tensor: np.expand_dims(image, 0)})

            output_image = output_image[0]
            for i in range(output_image.shape[2]):
                output_image[:, :, i] = minmax_scale(output_image[:, :, i])

            output_image = np.array(output_image, np.uint8)

            if label_path is not None:
                label_image_vis_gray = cv2.cvtColor(label_image_vis, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
                output_image_gray = cv2.cvtColor(output_image, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
                psnr = measure.compare_psnr(label_image_vis_gray, output_image_gray)
                ssim = measure.compare_ssim(label_image_vis_gray, output_image_gray)

                print('SSIM: {:.5f}'.format(ssim))
                print('PSNR: {:.5f}'.format(psnr))


            # 保存并可视化结果
            cv2.imwrite('src_img.png', image_vis)
            cv2.imwrite('derain_img.png', output_image)

            plt.figure('src_image')
            plt.imshow(image_vis[:, :, (2, 1, 0)])
            plt.figure('derain_ret')
            plt.imshow(output_image[:, :, (2, 1, 0)])
            plt.figure('atte_map_1')
            plt.axis('off')
            plt.imshow(atte_maps[0][0, :, :, 0])
            plt.savefig('atte_map_1.png', cmap='jet',bbox_inches='tight', dpi=300, pad_inches=0.0)
            plt.figure('atte_map_2')
            plt.axis('off')
            plt.imshow(atte_maps[1][0, :, :, 0])
            plt.savefig('atte_map_2.png', cmap='jet',bbox_inches='tight', dpi=300, pad_inches=0.0)
            plt.figure('atte_map_3')
            plt.axis('off')
            plt.imshow(atte_maps[2][0, :, :, 0])
            plt.savefig('atte_map_3.png', cmap='jet',bbox_inches='tight', dpi=300, pad_inches=0.0)
            plt.figure('atte_map_4')
            plt.axis('off')
            plt.imshow(atte_maps[3][0, :, :, 0])
            plt.savefig('atte_map_4.png', cmap='jet',bbox_inches='tight', dpi=300, pad_inches=0.0)
            plt.show()


if __name__ == '__main__':
    args = init_args()
    test_model(args.image_path, args.weights_path, args.label_path)
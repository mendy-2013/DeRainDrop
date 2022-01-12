# cited from:
#             @Author  : Luo Yao
#             @Site    : http://icode.baidu.com/repos/baidu/personal-code/Luoyao
#             @File    : derain_drop_net.py

import tensorflow as tf

from attentive_gan_model import attentive_gan_net
from attentive_gan_model import discriminative_net


class DeRainNet(object):

    def __init__(self, phase):
        self._phase = phase
        self._attentive_gan = attentive_gan_net.GenerativeNet(self._phase)
        self._discriminator = discriminative_net.DiscriminativeNet(self._phase)

    def compute_loss(self, input_tensor, gt_label_tensor, mask_label_tensor, name):
        with tf.variable_scope(name):
            # 计算attentive rnn loss
            attentive_rnn_loss, attentive_rnn_output = self._attentive_gan.compute_attentive_rnn_loss(
                input_tensor=input_tensor,
                label_tensor=mask_label_tensor,
                name='attentive_rnn_loss')

            auto_encoder_input = tf.concat((attentive_rnn_output, input_tensor), axis=-1)

            auto_encoder_loss, auto_encoder_output = self._attentive_gan.compute_autoencoder_loss(
                input_tensor=auto_encoder_input,
                label_tensor=gt_label_tensor,
                name='attentive_autoencoder_loss'
            )

            gan_loss = tf.add(attentive_rnn_loss, auto_encoder_loss)

            discriminative_inference, discriminative_loss = self._discriminator.compute_loss(
                input_tensor=auto_encoder_output,
                label_tensor=gt_label_tensor,
                attention_map=attentive_rnn_output,
                name='discriminative_loss')

            l_gan = tf.reduce_mean(tf.log(tf.subtract(tf.constant(1.0), discriminative_inference)) * 0.01)

            gan_loss = tf.add(gan_loss, l_gan)

            return gan_loss, discriminative_loss, auto_encoder_output

    # 用于测试
    def build(self, input_tensor, name):
        with tf.variable_scope(name):
            attentive_rnn_out = self._attentive_gan.build_attentive_rnn(
                input_tensor=input_tensor,
                name='attentive_rnn_loss/attentive_inference'
            )

            attentive_autoencoder_input = tf.concat((attentive_rnn_out['final_attention_map'],
                                                     input_tensor), axis=-1)

            output = self._attentive_gan.build_autoencoder(
                input_tensor=attentive_autoencoder_input,
                name='attentive_autoencoder_loss/autoencoder_inference'
            )

            return output['skip_3'], attentive_rnn_out['attention_map_list']

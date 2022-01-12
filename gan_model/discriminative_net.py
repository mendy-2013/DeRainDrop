# cited from:
#             @Author  : Luo Yao
#             @Site    : http://icode.baidu.com/repos/baidu/personal-code/Luoyao
#             @File    : discriminative_net.py

import tensorflow as tf

from attentive_gan_model import cnn_basenet


class DiscriminativeNet(cnn_basenet.CNNBaseModel):

    def __init__(self, phase):
        super(DiscriminativeNet, self).__init__()
        self._train_phase = tf.constant('train', dtype=tf.string)
        self._test_phase = tf.constant('test', dtype=tf.string)
        self._phase = phase
        self._is_training = self._init_phase()

    def _init_phase(self):
        return tf.equal(self._phase, self._train_phase)

    def _conv_stage(self, input_tensor, k_size, stride, out_dims, name):
        with tf.variable_scope(name):
            conv = self.conv2d(inputdata=input_tensor, out_channel=out_dims, kernel_size=k_size,
                               padding='SAME', stride=stride, use_bias=False, name='conv')

            relu = self.lrelu(conv, name='relu')

        return relu

    def build(self, input_tensor, name, reuse=False):
        with tf.variable_scope(name, reuse=reuse):
            conv_stage_1 = self._conv_stage(input_tensor=input_tensor, k_size=5,
                                            stride=1, out_dims=8,
                                            name='conv_stage_1')
            conv_stage_2 = self._conv_stage(input_tensor=conv_stage_1, k_size=5,
                                            stride=1, out_dims=16, name='conv_stage_2')
            conv_stage_3 = self._conv_stage(input_tensor=conv_stage_2, k_size=5,
                                            stride=1, out_dims=32, name='conv_stage_3')
            conv_stage_4 = self._conv_stage(input_tensor=conv_stage_3, k_size=5,
                                            stride=1, out_dims=64, name='conv_stage_4')
            conv_stage_5 = self._conv_stage(input_tensor=conv_stage_4, k_size=5,
                                            stride=1, out_dims=128, name='conv_stage_5')
            conv_stage_6 = self._conv_stage(input_tensor=conv_stage_5, k_size=5,
                                            stride=1, out_dims=128, name='conv_stage_6')
            attention_map = self.conv2d(inputdata=conv_stage_6, out_channel=1, kernel_size=5,
                                        padding='SAME', stride=1, use_bias=False, name='attention_map')
            conv_stage_7 = self._conv_stage(input_tensor=attention_map * conv_stage_6, k_size=5,
                                            stride=4, out_dims=64, name='conv_stage_7')
            conv_stage_8 = self._conv_stage(input_tensor=conv_stage_7, k_size=5,
                                            stride=4, out_dims=64, name='conv_stage_8')
            conv_stage_9 = self._conv_stage(input_tensor=conv_stage_8, k_size=5,
                                            stride=4, out_dims=32, name='conv_stage_9')
            fc_1 = self.fullyconnect(inputdata=conv_stage_9, out_dim=1024, use_bias=False, name='fc_1')
            fc_2 = self.fullyconnect(inputdata=fc_1, out_dim=1, use_bias=False, name='fc_2')
            fc_out = self.sigmoid(inputdata=fc_2, name='fc_out')

            fc_out = tf.where(tf.not_equal(fc_out, 1.0), fc_out, fc_out - 0.0000001)
            fc_out = tf.where(tf.not_equal(fc_out, 0.0), fc_out, fc_out + 0.0000001)

            return fc_out, attention_map, fc_2

    def compute_loss(self, input_tensor, label_tensor, attention_map, name):
        with tf.variable_scope(name):
            [batch_size, image_h, image_w, _] = input_tensor.get_shape().as_list()

            # 论文里的O
            zeros_mask = tf.zeros(shape=[batch_size, image_h, image_w, 1],
                                  dtype=tf.float32, name='O')
            fc_out_o, attention_mask_o, fc2_o = self.build(
                input_tensor=input_tensor, name='discriminative_inference')
            fc_out_r, attention_mask_r, fc2_r = self.build(
                input_tensor=label_tensor, name='discriminative_inference', reuse=True)

            l_map = tf.losses.mean_squared_error(attention_map, attention_mask_o) + \
                    tf.losses.mean_squared_error(attention_mask_r, zeros_mask)

            entropy_loss = -tf.log(fc_out_r) - tf.log(-tf.subtract(fc_out_o, tf.constant(1.0, tf.float32)))
            entropy_loss = tf.reduce_mean(entropy_loss)

            loss = entropy_loss + 0.05 * l_map

            return fc_out_o, loss

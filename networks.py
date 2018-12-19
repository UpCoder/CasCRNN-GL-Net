# -*- coding=utf-8 -*-
import tensorflow as tf
from tensorflow.contrib.slim.nets import resnet_v1
from tensorflow.contrib import slim
from models.research.slim.nets import resnet_v2, vgg, densenet_utils
import config


def parametric_relu(_x):
    alphas = tf.get_variable('alpha', _x.get_shape()[-1],
                             initializer=tf.constant_initializer(0.0),
                             dtype=tf.float32)
    pos = tf.nn.relu(_x)
    neg = alphas * (_x - abs(_x)) * 0.5
    return pos + neg


def conv_lstm(input, batch_size_ph, cell_output, final_output, cell_kernel, weight_decay, activation_fn=tf.nn.relu):
    '''

    :param input: 输入的tensor, 格式是[batch_size, step_size, h, w, channel]
    :param batch_size_ph: 输入的placeholder, 代表的是batch size用于初始化zero_state
    :param cell_output: ConvLSTMCell的输出
    :param cell_kernel: cell 里面卷积的kernel size
    :param final_output 3个cell的输出结合后再使用卷积的输出
    :return:
    '''
    batch_size, step_size, height, width, channel = input.get_shape().as_list()
    print(batch_size, step_size, height, width, channel )
    p_input_list = tf.split(input, step_size,
                            1)  # creates a list of leghth time_steps and one elemnt has the shape of (?, 400, 400, 1, 10)
    p_input_list = [tf.squeeze(p_input_, 1) for p_input_ in
                    p_input_list]  # remove the third dimention now one list elemnt has the shape of (?, 400, 400, 10)
    cell = tf.contrib.rnn.ConvLSTMCell(conv_ndims=2,  # ConvLSTMCell definition
                                       input_shape=[height, width, channel],
                                       output_channels=cell_output,
                                       kernel_shape=cell_kernel,
                                       skip_connection=False)
    print('batch_size_ph is ', batch_size_ph)
    state = cell.zero_state(batch_size_ph, dtype=tf.float32)
    outputs = []
    with tf.variable_scope("ConvLSTM") as scope:
        for i, p_input_ in enumerate(p_input_list):
            print('i is ', i, p_input_)
            if i > 0:
                scope.reuse_variables()
                print('set reuse')
            # ConvCell takes Tensor with size [batch_size, height, width, channel].
            t_output, state = cell(p_input_, state)
            outputs.append(t_output)
    outputs = tf.concat(outputs, axis=-1)
    print(outputs)
    with slim.arg_scope([slim.conv2d],
                        activation_fn=activation_fn,
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                        padding='SAME',
                        biases_initializer=tf.zeros_initializer()):
        _, _, _, output_channel = outputs.get_shape().as_list()
        outputs = slim.conv2d(outputs, output_channel, [3, 3], stride=1, scope='conv1')
        outputs = slim.conv2d(outputs, final_output, [1, 1], stride=1, scope='conv2')
    print(outputs)
    return outputs


class networks:
    def __init__(self, b_nc_roi, b_art_roi, b_pv_roi, b_nc_patch, b_art_patch, b_pv_patch, base_name, is_training,
                 num_classes, batch_size):
        self.roi_nc_input = b_nc_roi
        self.roi_art_input = b_art_roi
        self.roi_pv_input = b_pv_roi
        self.patch_nc_input = b_nc_patch
        self.patch_art_input = b_art_patch
        self.patch_pv_input = b_pv_patch
        self.batch_size = batch_size

        # patch_splited = tf.split(self.patch_input_tensor, num_or_size_splits=3, axis=-1)
        # self.patch_nc_input = tf.concat([patch_splited[0], patch_splited[0], patch_splited[0]], axis=-1)
        # self.patch_art_input = tf.concat([patch_splited[1], patch_splited[1], patch_splited[1]], axis=-1)
        # self.patch_pv_input = tf.concat([patch_splited[2], patch_splited[2], patch_splited[2]], axis=-1)
        # roi_splited = tf.split(self.roi_input_tensor, num_or_size_splits=3, axis=-1)
        # self.roi_nc_input = tf.concat([roi_splited[0], roi_splited[0], roi_splited[0]], axis=-1)
        # self.roi_art_input = tf.concat([roi_splited[1], roi_splited[1], roi_splited[1]], axis=-1)
        # self.roi_pv_input = tf.concat([roi_splited[2], roi_splited[2], roi_splited[2]], axis=-1)

        self.base_name = base_name
        self.is_training = is_training
        self.num_classes = num_classes
        self.build_base_network()
        # init the center feature
        _, dim = self.final_feature.get_shape().as_list()
        with tf.variable_scope('center'):
            self.centers = tf.get_variable('centers', [self.num_classes, dim], dtype=tf.float32,
                                           initializer=tf.constant_initializer(0),
                                           trainable=False)

    def build_base_network(self):
        print self.patch_nc_input
        patch_inputs = {
            'NC': self.patch_nc_input,
            'ART': self.patch_art_input,
            'PV': self.patch_pv_input
        }
        roi_inputs = {
            'NC': self.roi_nc_input,
            'ART': self.roi_art_input,
            'PV': self.roi_pv_input
        }
        phase_names_list = ['NC', 'ART', 'PV']
        patch_outputs = []
        roi_outputs = []
        if self.base_name == 'vgg16':
            with tf.variable_scope('patch_based'):
                for phase_idx, phase_name in enumerate(phase_names_list):
                    with slim.arg_scope(vgg.vgg_arg_scope()):
                        with slim.arg_scope([slim.conv2d, slim.fully_connected], reuse=(phase_idx != 0)):
                            print(phase_name, (phase_idx != 0), patch_inputs[phase_name])
                            outputs, end_points = vgg.vgg_16(patch_inputs[phase_name], 1000,
                                                             self.is_training, spatial_squeeze=False)
                            print end_points.keys()
                            patch_outputs.append(end_points['patch_based/vgg_16/fc8'])
            with tf.variable_scope('roi_based'):
                for phase_idx, phase_name in enumerate(phase_names_list):
                    with slim.arg_scope(vgg.vgg_arg_scope()):
                        with slim.arg_scope([slim.conv2d, slim.fully_connected], reuse=(phase_idx != 0)):
                            print phase_name, (phase_idx != 0)
                            outputs, end_points = vgg.vgg_16(roi_inputs[phase_name], 1000,
                                                             self.is_training, spatial_squeeze=False)
                            roi_outputs.append(end_points['roi_based/vgg_16/fc8'])
        elif self.base_name == 'res50':
            with tf.variable_scope('patch_based'):
                for phase_idx, phase_name in enumerate(phase_names_list):
                    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
                        with slim.arg_scope([slim.conv2d, slim.fully_connected], reuse=(phase_idx != 0)):
                            print(phase_name, (phase_idx != 0))
                            print('patch_inputs[phase_name] ', patch_inputs[phase_name])

                            outputs, end_points = resnet_v1.resnet_v1_50(patch_inputs[phase_name], None,
                                                                         self.is_training)
                            print end_points.keys()
                            patch_outputs.append(end_points['patch_based/resnet_v1_50/block4'])
            with tf.variable_scope('roi_based'):
                for phase_idx, phase_name in enumerate(phase_names_list):
                    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
                        with slim.arg_scope([slim.conv2d, slim.fully_connected], reuse=(phase_idx != 0)):
                            print phase_name, (phase_idx != 0)
                            outputs, end_points = resnet_v1.resnet_v1_50(patch_inputs[phase_name], None,
                                                                         self.is_training)
                            roi_outputs.append(end_points['roi_based/resnet_v1_50/block4'])
        else:
            print 'Keyword Error'
            assert False

        with tf.variable_scope('Global_Branch'):
            # gb_represent the global branch
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                activation_fn=tf.nn.relu,
                                weights_regularizer=slim.l2_regularizer(config.WEIGHT_DECAY),
                                biases_initializer=tf.zeros_initializer()):
                gb_rois = []
                gb_rate = 128
                for phase_idx, phase_name in enumerate(phase_names_list):
                    roi_output = slim.conv2d(roi_outputs[phase_idx], gb_rate, kernel_size=[3, 3],
                                             stride=1, scope=phase_name)
                    gb_rois.append(tf.expand_dims(roi_output, axis=1))
                gb_rois = tf.concat(gb_rois, axis=1)
                gb_roi = conv_lstm(gb_rois, self.batch_size, gb_rate // 2, gb_rate, [1, 1], config.WEIGHT_DECAY)
                gb_feature = tf.reduce_mean(gb_roi, axis=[1, 2])
                self.gb_logits = slim.fully_connected(gb_feature, self.num_classes, activation_fn=None)

        with tf.variable_scope('Local_Branch'):
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                activation_fn=tf.nn.relu,
                                weights_regularizer=slim.l2_regularizer(config.WEIGHT_DECAY),
                                biases_initializer=tf.zeros_initializer()):
                lb_rois = []
                lb_rate = 128
                for phase_idx, phase_name in enumerate(phase_names_list):
                    patch_output = slim.conv2d(patch_outputs[phase_idx], lb_rate, kernel_size=[3, 3], stride=1,
                                               scope=phase_name)
                    lb_rois.append(tf.expand_dims(patch_output, axis=1))
                lb_rois = tf.concat(lb_rois, axis=1)
                lb_roi = conv_lstm(lb_rois, self.batch_size, lb_rate // 2, lb_rate, [1, 1], config.WEIGHT_DECAY)
                lb_feature = tf.reduce_mean(lb_roi, axis=[1, 2])
                self.lb_logits = slim.fully_connected(lb_feature, self.num_classes, activation_fn=None)

        print('the roi_outputs is ', roi_outputs)
        print('the patch_outputs is ', patch_outputs)
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            weights_regularizer=slim.l2_regularizer(config.WEIGHT_DECAY),
                            biases_initializer=tf.zeros_initializer()):
            with slim.arg_scope([slim.conv2d], padding='SAME'):
                with tf.variable_scope('combining_patch_roi'):
                    # ART NC PV
                    art_gl_feature = tf.concat([patch_outputs[0], roi_outputs[0]], axis=-1)
                    nc_gl_feature = tf.concat([patch_outputs[1], roi_outputs[1]], axis=-1)
                    pv_gl_feature = tf.concat([patch_outputs[2], roi_outputs[2]], axis=-1)
                    art_gl_feature = slim.conv2d(art_gl_feature, 512, stride=1, kernel_size=[1, 1], scope='art')
                    nc_gl_feature = slim.conv2d(nc_gl_feature, 512, stride=1, kernel_size=[1, 1], scope='nc')
                    pv_gl_feature = slim.conv2d(pv_gl_feature, 512, stride=1, kernel_size=[1, 1], scope='pv')

                with tf.variable_scope('extracting_enhancement_pattern'):
                    triple_phase_feature = tf.concat([
                        tf.expand_dims(nc_gl_feature, axis=1),
                        tf.expand_dims(art_gl_feature, axis=1),
                        tf.expand_dims(pv_gl_feature, axis=1),
                    ], axis=1)
                    print('the triple_phase feature is ', triple_phase_feature)
                    final_feature = conv_lstm(triple_phase_feature, self.batch_size, 256, 512, [1, 1], config.WEIGHT_DECAY)
                with tf.variable_scope('classifing_fc'):
                    self.final_feature = tf.reduce_mean(final_feature, [1, 2])
                    print('final_featrue is ', self.final_feature)
                    logits = slim.fully_connected(self.final_feature, self.num_classes, activation_fn=None)
                    print('logits is ', logits)
                    self.logits = logits

    def update_centers(self, labels, alpha):
        with tf.variable_scope('center', reuse=True):
            centers = tf.get_variable('centers')

        labels = tf.reshape(labels, [-1])
        centers_batch = tf.gather(centers, labels)

        diff = centers_batch - self.final_feature

        # 获取一个batch中同一样本出现的次数，这里需要理解论文中的更新公式
        unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
        appear_times = tf.gather(unique_count, unique_idx)
        appear_times = tf.reshape(appear_times, [-1, 1])

        diff = diff / tf.cast((1 + appear_times), tf.float32)
        diff = alpha * diff
        # 更新中心
        centers = tf.scatter_sub(centers, labels, diff)

        return centers

    def build_loss(self, b_label, lambda_center_loss = 1.0, lambda_global_branch = 0.6, lambda_local_branch=0.2, add_to_collection=True):
        def _calculate_center_loss(features, labels):
            '''
            计算center loss
            :param features: B, C
            :param labels: B, C
            :return:
            '''
            with tf.variable_scope('center', reuse=True):
                centers = tf.get_variable('centers')

            len_features = features.get_shape()[1]
            labels = tf.reshape(labels, [-1])

            centers_batch = tf.gather(centers, labels)
            # 计算center loss的数值
            loss = tf.reduce_sum((features - centers_batch) ** 2, [1])

            return loss
        # assign the weight
        gt_tensor = tf.cast(b_label, tf.int32)
        class_weights = tf.constant([2.0, 2.0, 2.0, 2.0, 1.0])
        weights = tf.gather(class_weights, gt_tensor)
        print('build the loss')
        print('b_label is ', b_label)
        print('b_label', tf.squeeze(b_label, axis=1))
        print('logits is ', self.logits)
        # final_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
        #                                                                      labels=tf.squeeze(b_label, axis=1))
        final_cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=tf.squeeze(b_label, axis=1),
                                                                     logits=self.logits, weights=weights)
        final_cross_entropy_mean = tf.reduce_mean(final_cross_entropy)
        print('the final cross entropy mean is ', final_cross_entropy_mean)


        center_loss = _calculate_center_loss(self.final_feature, b_label)
        center_loss_mean = tf.reduce_mean(center_loss)
        print('center_loss_mean is ', center_loss_mean)

        # return final_cross_entropy_mean, center_loss_mean * lambda_center_loss

        # build for the global and local branches
        # global_cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.gb_logits,
        #                                                                                      labels=tf.squeeze(b_label,
        #                                                                                                        axis=1)))
        global_cross_entropy = tf.reduce_mean(
            tf.losses.sparse_softmax_cross_entropy(logits=self.gb_logits, labels=tf.squeeze(b_label, axis=1),
                                                   weights=weights))

        # local_cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.lb_logits,
        #                                                                                     labels=tf.squeeze(b_label,
        #                                                                                                       axis=1)))

        local_cross_entropy = tf.reduce_mean(
            tf.losses.sparse_softmax_cross_entropy(logits=self.lb_logits, weights=weights,
                                                   labels=tf.squeeze(b_label, axis=1)))

        if add_to_collection:
            tf.add_to_collection(tf.GraphKeys.LOSSES, local_cross_entropy * lambda_local_branch)
            tf.add_to_collection(tf.GraphKeys.LOSSES, global_cross_entropy * lambda_global_branch)
            tf.add_to_collection(tf.GraphKeys.LOSSES, center_loss_mean * lambda_center_loss)
            tf.add_to_collection(tf.GraphKeys.LOSSES, final_cross_entropy_mean)
            return final_cross_entropy_mean, center_loss_mean * lambda_center_loss, \
                   global_cross_entropy * lambda_global_branch, local_cross_entropy * lambda_local_branch, \
                   self.update_centers(b_label, 0.5)
        return final_cross_entropy_mean, center_loss_mean * lambda_center_loss, \
               global_cross_entropy * lambda_global_branch, local_cross_entropy * lambda_local_branch


class networks_with_attrs:
    def __init__(self, b_nc_roi, b_art_roi, b_pv_roi, b_nc_patch, b_art_patch, b_pv_patch, b_attrs, base_name,
                 is_training, num_classes, batch_size, use_attribute_flag=True, clstm_flag=True,
                 global_branch_flag=True, local_branch_flag=True):
        if global_branch_flag is False and local_branch_flag is False:
            print('global and local branch can not be False at the same time!')
            assert False
        self.roi_nc_input = b_nc_roi
        self.roi_art_input = b_art_roi
        self.roi_pv_input = b_pv_roi
        self.patch_nc_input = b_nc_patch
        self.patch_art_input = b_art_patch
        self.patch_pv_input = b_pv_patch
        self.attrs_input = b_attrs
        self.batch_size = batch_size
        self.use_attribute_flag = use_attribute_flag
        self.clstm_flag = clstm_flag
        self.global_branch_flag = global_branch_flag
        self.local_branch_flag = local_branch_flag

        self.base_name = base_name
        self.is_training = is_training
        self.num_classes = num_classes
        self.build_base_network()
        # init the center feature
        _, dim = self.final_feature.get_shape().as_list()
        with tf.variable_scope('center'):
            self.centers = tf.get_variable('centers', [self.num_classes, dim], dtype=tf.float32,
                                           initializer=tf.constant_initializer(0),
                                           trainable=False)

    def build_base_network(self):
        print self.patch_nc_input
        patch_inputs = {
            'NC': self.patch_nc_input,
            'ART': self.patch_art_input,
            'PV': self.patch_pv_input
        }
        roi_inputs = {
            'NC': self.roi_nc_input,
            'ART': self.roi_art_input,
            'PV': self.roi_pv_input
        }
        phase_names_list = ['NC', 'ART', 'PV']
        patch_outputs = []
        roi_outputs = []
        if self.base_name == 'vgg16':
            with tf.variable_scope('patch_based'):
                for phase_idx, phase_name in enumerate(phase_names_list):
                    with slim.arg_scope(vgg.vgg_arg_scope()):
                        with slim.arg_scope([slim.conv2d, slim.fully_connected], reuse=(phase_idx != 0)):
                            print(phase_name, (phase_idx != 0), patch_inputs[phase_name])
                            outputs, end_points = vgg.vgg_16(patch_inputs[phase_name], None,
                                                             self.is_training, spatial_squeeze=False, patch_flag=True)
                            print end_points.keys()
                            patch_outputs.append(end_points['final_feature'])
            with tf.variable_scope('roi_based'):
                for phase_idx, phase_name in enumerate(phase_names_list):
                    with slim.arg_scope(vgg.vgg_arg_scope()):
                        with slim.arg_scope([slim.conv2d, slim.fully_connected], reuse=(phase_idx != 0)):
                            print phase_name, (phase_idx != 0)
                            outputs, end_points = vgg.vgg_16(roi_inputs[phase_name], None,
                                                             self.is_training, spatial_squeeze=False,
                                                             fc_conv_padding='SAME')
                            roi_outputs.append(end_points['roi_based/vgg_16/fc7'])
        elif self.base_name == 'res50':
            if self.local_branch_flag:
                with tf.variable_scope('patch_based'):
                    for phase_idx, phase_name in enumerate(phase_names_list):
                        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
                            # with slim.arg_scope([slim.conv2d, slim.fully_connected], reuse=(phase_idx != 0)):
                            print(phase_name, (phase_idx != 0))
                            print('patch_inputs[phase_name] ', patch_inputs[phase_name])

                            outputs, end_points = resnet_v2.resnet_v2_50(patch_inputs[phase_name], None,
                                                                         self.is_training, patch_flag=True,
                                                                         global_pool=False,
                                                                         reuse=(phase_idx != 0))
                            print end_points.keys()
                            patch_outputs.append(end_points['final_feature'])
            if self.global_branch_flag:
                with tf.variable_scope('roi_based'):
                    for phase_idx, phase_name in enumerate(phase_names_list):
                        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
                            # with slim.arg_scope([slim.conv2d, slim.fully_connected], reuse=(phase_idx != 0)):
                            print phase_name, (phase_idx != 0)
                            outputs, end_points = resnet_v2.resnet_v2_50(roi_inputs[phase_name], None,
                                                                         self.is_training, reuse=(phase_idx != 0),
                                                                         global_pool=False,)
                            roi_outputs.append(end_points['final_feature'])
        elif self.base_name == 'dense121':
            if self.local_branch_flag:
                with tf.variable_scope('patch_based'):
                    for phase_idx, phase_name in enumerate(phase_names_list):
                        with slim.arg_scope(densenet_utils.densenet_arg_scope()):
                            # with slim.arg_scope([slim.conv2d, slim.fully_connected], reuse=(phase_idx != 0)):
                            print(phase_name, (phase_idx != 0))
                            print('patch_inputs[phase_name] ', patch_inputs[phase_name])
                            outputs, end_points = densenet_utils.densenet121(patch_inputs[phase_name], None,
                                                                             is_training=self.is_training,
                                                                             is_patch=True, reuse=(phase_idx != 0))
                            print end_points.keys()
                            patch_outputs.append(end_points['final_feature'])
            if self.global_branch_flag:
                with tf.variable_scope('roi_based'):
                    for phase_idx, phase_name in enumerate(phase_names_list):
                        with slim.arg_scope(densenet_utils.densenet_arg_scope()):
                            # with slim.arg_scope([slim.conv2d, slim.fully_connected], reuse=(phase_idx != 0)):
                            print phase_name, (phase_idx != 0)
                            outputs, end_points = densenet_utils.densenet121(roi_inputs[phase_name], None,
                                                                             is_training=self.is_training,
                                                                             is_patch=False, reuse=(phase_idx != 0))
                            roi_outputs.append(end_points['final_feature'])
        else:
            print 'Keyword Error'
            assert False
        if self.use_attribute_flag:
            print('without attribute feature')
            with tf.variable_scope('Attribute_Feature'):
                with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                    activation_fn=tf.nn.relu,
                                    weights_regularizer=slim.l2_regularizer(config.WEIGHT_DECAY),
                                    biases_initializer=tf.zeros_initializer()):
                    num_attr_feature = 32
                    attrs_input = tf.expand_dims(tf.expand_dims(self.attrs_input, axis=1), axis=1)
                    attribute_feature = slim.conv2d(attrs_input, num_attr_feature, kernel_size=[1, 1], stride=1,
                                                    scope='conv1')
                    self.attribute_feature = slim.conv2d(attribute_feature, num_attr_feature, kernel_size=[1, 1],
                                                         stride=1, scope='conv2')

        # 2048 for Resnet50
        # 4096 for VGG16
        if self.global_branch_flag:
            with tf.variable_scope('Global_Branch'):
                # gb_represent the global branch
                with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                    activation_fn=tf.nn.relu,
                                    weights_regularizer=slim.l2_regularizer(config.WEIGHT_DECAY),
                                    biases_initializer=tf.zeros_initializer()):
                    gb_rois = []
                    gb_rate = 256
                    for phase_idx, phase_name in enumerate(phase_names_list):
                        roi_output = slim.conv2d(roi_outputs[phase_idx], gb_rate, kernel_size=[3, 3],
                                                 stride=1, scope=phase_name)
                        gb_rois.append(tf.expand_dims(roi_output, axis=1))
                    if self.clstm_flag:
                        gb_rois = tf.concat(gb_rois, axis=1)
                        gb_roi = conv_lstm(gb_rois, self.batch_size, gb_rate // 2, gb_rate, [1, 1], config.WEIGHT_DECAY,
                                           activation_fn=parametric_relu)
                    else:
                        print('gb rois is ', gb_rois)
                        gb_roi = tf.squeeze(tf.concat(gb_rois, axis=-1), axis=1)
                        print('gb roi is ', gb_roi)
                        gb_roi = slim.conv2d(gb_roi, gb_rate, kernel_size=[1, 1], stride=1, activation_fn=tf.nn.relu,
                                             scope='inter-phase-feature')
                    gb_feature = tf.reduce_mean(gb_roi, axis=[1, 2])
                    if self.use_attribute_flag:
                        gb_feature = tf.concat([gb_feature, tf.squeeze(self.attribute_feature, axis=[1, 2])], axis=-1)
                    self.gb_logits = slim.fully_connected(gb_feature,
                                                          self.num_classes, activation_fn=None, scope='logits_layer')
        # 512 for resnet50
        # 512 for vgg16
        if self.local_branch_flag:
            with tf.variable_scope('Local_Branch'):
                with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                    activation_fn=tf.nn.relu,
                                    weights_regularizer=slim.l2_regularizer(config.WEIGHT_DECAY),
                                    biases_initializer=tf.zeros_initializer()):
                    lb_rois = []
                    lb_rate = 128
                    for phase_idx, phase_name in enumerate(phase_names_list):
                        patch_output = slim.conv2d(patch_outputs[phase_idx], lb_rate, kernel_size=[3, 3], stride=1,
                                                   scope=phase_name)
                        lb_rois.append(tf.expand_dims(patch_output, axis=1))
                    if self.clstm_flag:
                        lb_rois = tf.concat(lb_rois, axis=1)
                        lb_roi = conv_lstm(lb_rois, self.batch_size, lb_rate // 2, lb_rate, [1, 1], config.WEIGHT_DECAY,
                                           activation_fn=parametric_relu)
                    else:
                        print 'lb rois are ', lb_rois
                        lb_roi = tf.squeeze(tf.concat(lb_rois, axis=-1), axis=1)
                        lb_roi = slim.conv2d(lb_roi, lb_rate, kernel_size=[1, 1], stride=1, activation_fn=tf.nn.relu,
                                             scope='inter-phase-feature')
                        print 'lb roi is ', lb_roi
                    lb_feature = tf.reduce_mean(lb_roi, axis=[1, 2])
                    if self.use_attribute_flag:
                        lb_feature = tf.concat([lb_feature, tf.squeeze(self.attribute_feature, axis=[1, 2])], axis=-1)
                    self.lb_logits = slim.fully_connected(lb_feature,
                                                          self.num_classes, activation_fn=None, scope='logits_layer')

        print('the roi_outputs is   ', roi_outputs)
        print('the patch_outputs is ', patch_outputs)
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            weights_regularizer=slim.l2_regularizer(config.WEIGHT_DECAY),
                            biases_initializer=tf.zeros_initializer()):
            with slim.arg_scope([slim.conv2d], padding='SAME'):
                if self.global_branch_flag and self.local_branch_flag:
                    with tf.variable_scope('combining_patch_roi'):
                        final_intra_phase_rate = 512
                        # ART NC PV
                        print('patch_outputs is ', patch_outputs[0])
                        print('roi_outputs is ', roi_outputs[0])
                        art_gl_feature = tf.concat([patch_outputs[0], roi_outputs[0]], axis=-1)
                        nc_gl_feature = tf.concat([patch_outputs[1], roi_outputs[1]], axis=-1)
                        pv_gl_feature = tf.concat([patch_outputs[2], roi_outputs[2]], axis=-1)
                        art_gl_feature = slim.conv2d(art_gl_feature, final_intra_phase_rate, stride=1, kernel_size=[1, 1],
                                                     scope='art')
                        nc_gl_feature = slim.conv2d(nc_gl_feature, final_intra_phase_rate, stride=1, kernel_size=[1, 1],
                                                    scope='nc')
                        pv_gl_feature = slim.conv2d(pv_gl_feature, final_intra_phase_rate, stride=1, kernel_size=[1, 1],
                                                    scope='pv')
                    with tf.variable_scope('extracting_enhancement_pattern'):
                        triple_phase_feature = tf.concat([
                            tf.expand_dims(nc_gl_feature, axis=1),
                            tf.expand_dims(art_gl_feature, axis=1),
                            tf.expand_dims(pv_gl_feature, axis=1),
                        ], axis=1)
                        print('the triple_phase feature is ', triple_phase_feature)
                        final_inter_phase_rate = 256
                        if self.clstm_flag:
                            final_feature = conv_lstm(triple_phase_feature, self.batch_size, final_inter_phase_rate // 2,
                                                      final_inter_phase_rate, [1, 1], config.WEIGHT_DECAY,
                                                      activation_fn=parametric_relu)
                        else:

                            final_feature = tf.concat([nc_gl_feature, art_gl_feature, pv_gl_feature], axis=-1)
                            final_feature = slim.conv2d(final_feature, final_inter_phase_rate, kernel_size=[1, 1], stride=1,
                                                        scope='inter-phase-feature', activation_fn=tf.nn.relu)
                elif self.global_branch_flag:
                    with tf.variable_scope('extracting_enhancement_pattern'):
                        triple_phase_feature = tf.concat([
                            tf.expand_dims(roi_outputs[0], axis=1),
                            tf.expand_dims(roi_outputs[1], axis=1),
                            tf.expand_dims(roi_outputs[2], axis=1),
                        ], axis=1)
                        final_inter_phase_rate = 256
                        final_feature = conv_lstm(triple_phase_feature, self.batch_size, final_inter_phase_rate // 2,
                                                  final_inter_phase_rate, [1, 1], config.WEIGHT_DECAY,
                                                  activation_fn=parametric_relu)
                elif self.local_branch_flag:
                    with tf.variable_scope('extracting_enhancement_pattern'):
                        triple_phase_feature = tf.concat([
                            tf.expand_dims(patch_outputs[0], axis=1),
                            tf.expand_dims(patch_outputs[1], axis=1),
                            tf.expand_dims(patch_outputs[2], axis=1),
                        ], axis=1)
                        final_inter_phase_rate = 256
                        final_feature = conv_lstm(triple_phase_feature, self.batch_size, final_inter_phase_rate // 2,
                                                  final_inter_phase_rate, [1, 1], config.WEIGHT_DECAY,
                                                  activation_fn=parametric_relu)
                else:
                    print('can not generate final feature output!')
                    assert False
                with tf.variable_scope('classifing_fc'):
                    self.final_feature = tf.reduce_mean(final_feature, [1, 2])
                    print('final_featrue is ', self.final_feature)
                    if self.use_attribute_flag:
                        cls_feature = tf.concat(
                            [tf.expand_dims(tf.expand_dims(self.final_feature, axis=1), axis=1), self.attribute_feature],
                            axis=-1)
                        logits = slim.conv2d(cls_feature, self.num_classes, kernel_size=[1, 1], stride=1,
                                             activation_fn=None, scope='logits_layer')
                    else:
                        logits = slim.conv2d(tf.expand_dims(tf.expand_dims(self.final_feature, axis=1), axis=1),
                                             self.num_classes, kernel_size=[1, 1], stride=1, activation_fn=None,
                                             scope='logits_layer')
                    self.logits = tf.reduce_mean(logits, [1, 2])

    def update_centers(self, labels, alpha):
        '''
        更新center
        :param labels: Batch, 不是one hot格式编码
        :param alpha: float, 学习率
        :return:
        '''
        with tf.variable_scope('center', reuse=True):
            centers = tf.get_variable('centers')

        # labels = tf.reshape(labels, [-1])
        centers_batch = tf.gather(centers, labels)

        diff = centers_batch - self.final_feature

        # 获取一个batch中同一样本出现的次数，这里需要理解论文中的更新公式
        unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
        appear_times = tf.gather(unique_count, unique_idx)
        appear_times = tf.reshape(appear_times, [-1, 1])

        diff = diff / tf.cast((1 + appear_times), tf.float32)
        diff = alpha * diff
        # 更新中心
        centers = tf.scatter_sub(centers, labels, diff)

        return centers

    def build_loss(self, b_label, lambda_center_loss=1.0, lambda_global_branch=0.8, lambda_local_branch=0.2,
                   alpha=0.5, add_to_collection=True):
        def _calculate_center_loss(features, labels):
            '''
            计算center loss
            :param features: B, C
            :param labels: B
            :return:
            '''
            with tf.variable_scope('center', reuse=True):
                centers = tf.get_variable('centers')

            len_features = features.get_shape()[1]
            labels = tf.reshape(labels, [-1])

            centers_batch = tf.gather(centers, labels)
            # 计算center loss的数值
            loss = tf.reduce_sum((features - centers_batch) ** 2, [1])

            return loss

        gt_tensor = tf.cast(b_label, tf.int32)
        # class_weights = tf.constant([2.0, 2.0, 2.0, 2.0, 1.0])
        class_weights = tf.constant([1.0, 1.0, 1.0, 1.0, 1.0])
        weights = tf.gather(class_weights, gt_tensor)
        print('build the loss')
        print('b_label is ', b_label)
        print('b_label', tf.squeeze(b_label, axis=1))
        print('logits is ', self.logits)
        # final_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
        #                                                                      labels=tf.squeeze(b_label, axis=1))
        # final_cross_entropy_mean = tf.reduce_mean(final_cross_entropy)
        final_cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=tf.squeeze(b_label, axis=1),
                                                                     loss_collection=None, logits=self.logits,
                                                                     weights=weights)
        final_cross_entropy_mean = tf.reduce_mean(final_cross_entropy)
        print('the final cross entropy mean is ', final_cross_entropy_mean)

        center_loss = _calculate_center_loss(self.final_feature, tf.squeeze(b_label, axis=1))
        center_loss_mean = tf.reduce_mean(center_loss)
        print('center_loss_mean is ', center_loss_mean)

        # build for the global and local branches
        global_cross_entropy = tf.Variable(0.0, trainable=False)
        if self.global_branch_flag:
            # global_cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.gb_logits,
            #                                                                                      labels=tf.squeeze(b_label,
            #                                                                                       axis=1)))

            global_cross_entropy = tf.reduce_mean(
                tf.losses.sparse_softmax_cross_entropy(logits=self.gb_logits, labels=tf.squeeze(b_label, axis=1),
                                                       weights=weights, loss_collection=None,))
        local_cross_entropy = tf.Variable(0.0, trainable=False)
        if self.local_branch_flag:
            # local_cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.lb_logits,
            #                                                                                     labels=tf.squeeze(b_label,
            #                                                                                                 axis=1)))
            local_cross_entropy = tf.reduce_mean(
                tf.losses.sparse_softmax_cross_entropy(logits=self.lb_logits, weights=weights, loss_collection=None,
                                                       labels=tf.squeeze(b_label, axis=1)))
        if add_to_collection:
            if self.global_branch_flag:
                tf.add_to_collection(tf.GraphKeys.LOSSES, global_cross_entropy * lambda_global_branch)
            tf.add_to_collection(tf.GraphKeys.LOSSES, center_loss_mean * lambda_center_loss)
            tf.add_to_collection(tf.GraphKeys.LOSSES, final_cross_entropy_mean)
            if self.local_branch_flag:
                tf.add_to_collection(tf.GraphKeys.LOSSES, local_cross_entropy * lambda_local_branch)
            return final_cross_entropy_mean, center_loss_mean * lambda_center_loss, \
                   global_cross_entropy * lambda_global_branch, local_cross_entropy * lambda_local_branch, \
                   self.update_centers(tf.squeeze(b_label, axis=1), alpha=alpha)
        else:
            return final_cross_entropy_mean, center_loss_mean * lambda_center_loss, \
                   global_cross_entropy * lambda_global_branch, local_cross_entropy * lambda_local_branch


if __name__ == '__main__':
    input_tensor = tf.placeholder(tf.float32, [None, 224, 224, 3], name='input')
    is_training = tf.placeholder(tf.bool, [], name='is_training')
    networks(input_tensor, input_tensor, input_tensor, input_tensor, input_tensor, input_tensor, 'vgg16', is_training,
             4, 16)

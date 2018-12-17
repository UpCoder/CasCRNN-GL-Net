# -*- coding=utf-8 -*-
import tensorflow as tf
import math
import config
from networks import networks, networks_with_attrs
import os
import numpy as np
from glob import glob
import cv2
import util
import random
from dataset.medicalImage import read_mhd_image, read_liver_mask, get_filled_mask, get_liver_path
from dataset.convert2jpg import extract_patches, calculate_mask_attributes


def preprocessing_image(image, output_height, output_width):
    # _R_MEAN = 164.0
    # _G_MEAN = 164.0
    # _B_MEAN = 164.0
    resized_image = cv2.resize(image, (output_height, output_width))
    resized_image = np.asarray(resized_image, np.float32)
    # resized_image -= [_R_MEAN, _G_MEAN, _B_MEAN]
    return resized_image


class Generate_Batch_Data_v2:
    def __init__(self, nc_rois, art_rois, pv_rois, nc_patches, art_patches, pv_patches, batch_size, labels=None):
        self.nc_rois = nc_rois
        self.art_rois = art_rois
        self.pv_rois = pv_rois
        self.nc_patches = nc_patches
        self.art_patches = art_patches
        self.pv_patches = pv_patches
        self.batch_size = batch_size
        self.labels = labels
        self.start_index = 0

    def generate_next_batch(self):
        while self.start_index < len(self.nc_rois):
            epoch_end = False
            end_index = self.start_index + self.batch_size
            if end_index > len(self.nc_rois):
                end_index = len(self.nc_rois)
            nc_batch_rois = self.nc_rois[self.start_index: end_index]
            art_batch_rois = self.art_rois[self.start_index: end_index]
            pv_batch_rois = self.pv_rois[self.start_index: end_index]
            nc_batch_patches = self.nc_patches[self.start_index: end_index]
            art_batch_patches = self.art_patches[self.start_index: end_index]
            pv_batch_patches = self.pv_patches[self.start_index: end_index]
            # nc_batch_rois = [cv2.imread(nc_batch_roi_path) for nc_batch_roi_path in nc_batch_roi_paths]
            # art_batch_rois = [cv2.imread(art_batch_roi_path) for art_batch_roi_path in art_batch_roi_paths]
            # pv_batch_rois = [cv2.imread(pv_batch_roi_path) for pv_batch_roi_path in pv_batch_roi_paths]
            # art_batch_patches = [cv2.imread(art_patch_path) for art_patch_path in art_batch_patch_paths]
            # nc_batch_patches = [cv2.imread(nc_patch_path) for nc_patch_path in nc_batch_patch_paths]
            # pv_batch_patches = [cv2.imread(pv_patch_path) for pv_patch_path in pv_batch_patch_paths]
            nc_batch_rois_preprocessed = [
                preprocessing_image(nc_roi, config.ROI_IMAGE_HEIGHT, config.ROI_IMAGE_WIDTH) for nc_roi in
                nc_batch_rois]
            art_batch_rois_preprocessed = [
                preprocessing_image(art_roi, config.ROI_IMAGE_HEIGHT, config.ROI_IMAGE_WIDTH) for art_roi in
                art_batch_rois]
            pv_batch_rois_preprocessed = [
                preprocessing_image(pv_roi, config.ROI_IMAGE_HEIGHT, config.ROI_IMAGE_WIDTH) for pv_roi in
                pv_batch_rois]

            nc_batch_patches_preprocessed = [
                preprocessing_image(nc_patch, config.ROI_IMAGE_HEIGHT, config.ROI_IMAGE_WIDTH) for nc_patch in
                nc_batch_patches
            ]
            art_batch_patches_preprocessed = [
                preprocessing_image(art_patch, config.ROI_IMAGE_HEIGHT, config.ROI_IMAGE_WIDTH) for art_patch in
                art_batch_patches
            ]
            pv_batch_patches_preprocessed = [
                preprocessing_image(pv_patch, config.ROI_IMAGE_HEIGHT, config.ROI_IMAGE_WIDTH) for pv_patch in
                pv_batch_patches
            ]
            if self.labels is not None:
                batch_labels = self.labels[self.start_index: end_index]
            self.start_index = end_index
            if end_index == len(self.nc_rois):
                self.start_index = 0
                epoch_end = True
            if self.labels is not None:
                yield nc_batch_rois_preprocessed, art_batch_rois_preprocessed, pv_batch_rois_preprocessed, \
                      nc_batch_patches_preprocessed, art_batch_patches_preprocessed, pv_batch_patches_preprocessed, \
                      batch_labels, epoch_end
            else:
                yield nc_batch_rois_preprocessed, art_batch_rois_preprocessed, pv_batch_rois_preprocessed, \
                      nc_batch_patches_preprocessed, art_batch_patches_preprocessed, pv_batch_patches_preprocessed, \
                      epoch_end


class Generate_Batch_Data_v2_with_attributes:
    def __init__(self, nc_rois, art_rois, pv_rois, nc_patches, art_patches, pv_patches, nc_attrs, art_attrs, pv_attrs,
                 batch_size, labels=None):
        self.nc_rois = nc_rois
        self.art_rois = art_rois
        self.pv_rois = pv_rois
        self.nc_patches = nc_patches
        self.art_patches = art_patches
        self.pv_patches = pv_patches
        self.batch_size = batch_size
        self.nc_attrs = np.asarray(nc_attrs, np.float32)
        self.art_attrs = np.asarray(art_attrs, np.float32)
        self.pv_attrs = np.asarray(pv_attrs, np.float32)
        self.labels = labels
        self.start_index = 0

    def generate_next_batch(self):
        while self.start_index < len(self.nc_rois):
            epoch_end = False
            end_index = self.start_index + self.batch_size
            if end_index > len(self.nc_rois):
                end_index = len(self.nc_rois)
            nc_batch_rois = self.nc_rois[self.start_index: end_index]
            art_batch_rois = self.art_rois[self.start_index: end_index]
            pv_batch_rois = self.pv_rois[self.start_index: end_index]
            nc_batch_patches = self.nc_patches[self.start_index: end_index]
            art_batch_patches = self.art_patches[self.start_index: end_index]
            pv_batch_patches = self.pv_patches[self.start_index: end_index]
            nc_batch_attrs = self.nc_attrs[self.start_index: end_index]
            art_batch_attrs = self.art_attrs[self.start_index: end_index]
            pv_batch_attrs = self.pv_attrs[self.start_index: end_index]
            # nc_batch_rois = [cv2.imread(nc_batch_roi_path) for nc_batch_roi_path in nc_batch_roi_paths]
            # art_batch_rois = [cv2.imread(art_batch_roi_path) for art_batch_roi_path in art_batch_roi_paths]
            # pv_batch_rois = [cv2.imread(pv_batch_roi_path) for pv_batch_roi_path in pv_batch_roi_paths]
            # art_batch_patches = [cv2.imread(art_patch_path) for art_patch_path in art_batch_patch_paths]
            # nc_batch_patches = [cv2.imread(nc_patch_path) for nc_patch_path in nc_batch_patch_paths]
            # pv_batch_patches = [cv2.imread(pv_patch_path) for pv_patch_path in pv_batch_patch_paths]
            nc_batch_rois_preprocessed = [
                preprocessing_image(nc_roi, config.ROI_IMAGE_HEIGHT, config.ROI_IMAGE_WIDTH) for nc_roi in
                nc_batch_rois]
            art_batch_rois_preprocessed = [
                preprocessing_image(art_roi, config.ROI_IMAGE_HEIGHT, config.ROI_IMAGE_WIDTH) for art_roi in
                art_batch_rois]
            pv_batch_rois_preprocessed = [
                preprocessing_image(pv_roi, config.ROI_IMAGE_HEIGHT, config.ROI_IMAGE_WIDTH) for pv_roi in
                pv_batch_rois]

            nc_batch_patches_preprocessed = [
                preprocessing_image(nc_patch, config.PATCH_IMAGE_HEIGHT, config.PATCH_IMAGE_WIDTH) for nc_patch in
                nc_batch_patches
            ]
            art_batch_patches_preprocessed = [
                preprocessing_image(art_patch, config.PATCH_IMAGE_HEIGHT, config.PATCH_IMAGE_WIDTH) for art_patch in
                art_batch_patches
            ]
            pv_batch_patches_preprocessed = [
                preprocessing_image(pv_patch, config.PATCH_IMAGE_HEIGHT, config.PATCH_IMAGE_WIDTH) for pv_patch in
                pv_batch_patches
            ]
            nc_batch_attrs /= config.MAX_ATTRS
            art_batch_attrs /= config.MAX_ATTRS
            pv_batch_attrs /= config.MAX_ATTRS
            cur_attrs = np.concatenate([nc_batch_attrs, art_batch_attrs, pv_batch_attrs], axis=-1)
            if self.labels is not None:
                batch_labels = self.labels[self.start_index: end_index]
            self.start_index = end_index
            if end_index == len(self.nc_rois):
                self.start_index = 0
                epoch_end = True
            if self.labels is not None:
                yield nc_batch_rois_preprocessed, art_batch_rois_preprocessed, pv_batch_rois_preprocessed, \
                      nc_batch_patches_preprocessed, art_batch_patches_preprocessed, pv_batch_patches_preprocessed, \
                      cur_attrs, batch_labels, epoch_end
            else:
                yield nc_batch_rois_preprocessed, art_batch_rois_preprocessed, pv_batch_rois_preprocessed, \
                      nc_batch_patches_preprocessed, art_batch_patches_preprocessed, pv_batch_patches_preprocessed, \
                      cur_attrs, epoch_end


def generate_roi_feature_with_attributions(cur_dataset_dir, slice_name, patch_size, sess, logits, nc_roi_placeholder, art_roi_placeholder,
                         pv_roi_placeholder, nc_patch_placeholder, art_patch_placeholder, pv_patch_placeholder, attrs_placeholder,
                         batch_size_placeholder):

    cur_slice_dir = os.path.join(cur_dataset_dir, slice_name)

    nc_img_path = glob(os.path.join(cur_slice_dir, 'NC_Image*.mhd'))[0]
    art_img_path = glob(os.path.join(cur_slice_dir, 'ART_Image*.mhd'))[0]
    pv_img_path = glob(os.path.join(cur_slice_dir, 'PV_Image*.mhd'))[0]
    nc_img = np.squeeze(read_mhd_image(nc_img_path))
    art_img = np.squeeze(read_mhd_image(art_img_path))
    pv_img = np.squeeze(read_mhd_image(pv_img_path))
    nc_mask_path = glob(os.path.join(cur_slice_dir, 'NC_Mask*.mhd'))[0]
    art_mask_path = glob(os.path.join(cur_slice_dir, 'ART_Mask*.mhd'))[0]
    pv_mask_path = glob(os.path.join(cur_slice_dir, 'PV_Mask*.mhd'))[0]
    nc_mask = np.squeeze(read_mhd_image(nc_mask_path))
    art_mask = np.squeeze(read_mhd_image(art_mask_path))
    pv_mask = np.squeeze(read_mhd_image(pv_mask_path))

    nc_mask, nc_min_xs, nc_min_ys, nc_max_xs, nc_max_ys = get_filled_mask(nc_mask)
    art_mask, art_min_xs, art_min_ys, art_max_xs, art_max_ys = get_filled_mask(art_mask)
    pv_mask, pv_min_xs, pv_min_ys, pv_max_xs, pv_max_ys = get_filled_mask(pv_mask)
    height_nc, width_nc, perimeter_nc, area_nc, circle_metric_nc, edge_nc = calculate_mask_attributes(nc_mask)
    nc_attr = [height_nc, width_nc, perimeter_nc, area_nc, circle_metric_nc]
    height_art, width_art, perimeter_art, area_art, circle_metric_art, edge_art = calculate_mask_attributes(
        art_mask)
    art_attr = [height_art, width_art, perimeter_art, area_art, circle_metric_art]
    height_pv, width_pv, perimeter_pv, area_pv, circle_metric_pv, edge_pv = calculate_mask_attributes(pv_mask)
    pv_attr = [height_pv, width_pv, perimeter_pv, area_pv, circle_metric_pv]

    nc_liver_mask_path = get_liver_path(cur_slice_dir, 'nc')
    art_liver_mask_path = get_liver_path(cur_slice_dir, 'art')
    pv_liver_mask_path = get_liver_path(cur_slice_dir, 'pv')
    nc_liver_min_xs, nc_liver_min_ys, nc_liver_max_xs, nc_liver_max_ys = read_liver_mask(nc_liver_mask_path)
    art_liver_min_xs, art_liver_min_ys, art_liver_max_xs, art_liver_max_ys = read_liver_mask(art_liver_mask_path)
    pv_liver_min_xs, pv_liver_min_ys, pv_liver_max_xs, pv_liver_max_ys = read_liver_mask(pv_liver_mask_path)
    nc_liver = nc_img[nc_liver_min_xs: nc_liver_max_xs, nc_liver_min_ys: nc_liver_max_ys]
    art_liver = art_img[art_liver_min_xs: art_liver_max_xs, art_liver_min_ys: art_liver_max_ys]
    pv_liver = pv_img[pv_liver_min_xs: pv_liver_max_xs, pv_liver_min_ys: pv_liver_max_ys]

    # 非病灶区域置0
    nc_roi_img = np.copy(nc_img)
    art_roi_img = np.copy(art_img)
    pv_roi_img = np.copy(pv_img)
    nc_roi_img[nc_mask != 1] = 0
    art_roi_img[art_mask != 1] = 0
    pv_roi_img[pv_mask != 1] = 0
    nc_roi_img = nc_roi_img[nc_min_xs: nc_max_xs, nc_min_ys: nc_max_ys]
    art_roi_img = art_roi_img[art_min_xs: art_max_xs, art_min_ys: art_max_ys]
    pv_roi_img = pv_roi_img[pv_min_xs: pv_max_xs, pv_min_ys: pv_max_ys]

    nc_h, nc_w = np.shape(nc_roi_img)
    art_h, art_w = np.shape(art_roi_img)
    pv_h, pv_w = np.shape(pv_roi_img)
    nc_size = nc_h * nc_w
    art_size = art_h * art_w
    pv_size = pv_h * pv_w
    if pv_size >= nc_size and pv_size >= art_size:
        target_h = pv_h
        target_w = pv_w
    elif art_size >= nc_size and art_size >= pv_size:
        target_h = art_h
        target_w = art_w
    elif nc_size >= art_size and nc_size >= pv_size:
        target_h = nc_h
        target_w = nc_w
    if target_h < patch_size:
        target_h = patch_size + 1
    if target_w < patch_size:
        target_w = patch_size + 1
    nc_roi_img = np.asarray(nc_roi_img, np.float32)
    art_roi_img = np.asarray(art_roi_img, np.float32)
    pv_roi_img = np.asarray(pv_roi_img, np.float32)
    nc_liver = np.asarray(nc_liver, np.float32)
    art_liver = np.asarray(art_liver, np.float32)
    pv_liver = np.asarray(pv_liver, np.float32)
    nc_roi_resized = cv2.resize(nc_roi_img, (target_h, target_w))
    art_roi_resized = cv2.resize(art_roi_img, (target_h, target_w))
    pv_roi_resized = cv2.resize(pv_roi_img, (target_h, target_w))
    nc_liver_resized = cv2.resize(nc_liver, (target_h, target_w))
    art_liver_resized = cv2.resize(art_liver, (target_h, target_w))
    pv_liver_resized = cv2.resize(pv_liver, (target_h, target_w))

    nc_roi_final = np.concatenate([np.expand_dims(nc_roi_resized, axis=2), np.expand_dims(nc_liver_resized, axis=2),
                                   np.expand_dims(nc_roi_resized, axis=2)], axis=2)
    art_roi_final = np.concatenate(
        [np.expand_dims(art_roi_resized, axis=2), np.expand_dims(art_liver_resized, axis=2),
         np.expand_dims(art_roi_resized, axis=2)], axis=2)
    pv_roi_final = np.concatenate([np.expand_dims(pv_roi_resized, axis=2), np.expand_dims(pv_liver_resized, axis=2),
                                   np.expand_dims(pv_roi_resized, axis=2)], axis=2)
    cur_nc_patches, cur_art_patches, cur_pv_patches = extract_patches(nc_roi_final, art_roi_final, pv_roi_final,
                                                                      patch_size=patch_size)

    print('nc mean is ', nc_roi_final.mean((0, 1)))
    print('art mean is ', art_roi_final.mean((0, 1)))
    print('pv mean is ', pv_roi_final.mean((0, 1)))
    print(
        'size of roi is ', np.shape(nc_roi_final), np.shape(art_roi_final), np.shape(pv_roi_final), len(cur_nc_patches))
    if len(cur_nc_patches) == 0:
        print('the number of patches is zero')
        assert False
    nc_rois = [nc_roi_final] * len(cur_nc_patches)
    art_rois = [art_roi_final] * len(cur_art_patches)
    pv_rois = [pv_roi_final] * len(cur_pv_patches)
    nc_attrs = [nc_attr] * len(cur_nc_patches)
    art_attrs = [art_attr] * len(cur_art_patches)
    pv_attrs = [pv_attr] * len(cur_pv_patches)

    labels = [int(slice_name[-1])] * len(cur_pv_patches)
    dataset = Generate_Batch_Data_v2_with_attributes(nc_rois, art_rois, pv_rois, cur_nc_patches, cur_art_patches,
                                                     cur_pv_patches, nc_attrs, art_attrs, pv_attrs,
                                                     config.val_batch_size, labels)
    generator = dataset.generate_next_batch()
    logits_values = []
    batch_count = 1
    while True:
        nc_batch_rois, art_batch_rois, pv_batch_rois, \
        nc_batch_patches, art_batch_patches, pv_batch_patches, \
        attrs, batch_label, epoch_end = generator.next()
        feed_dict = {
            nc_roi_placeholder: nc_batch_rois,
            art_roi_placeholder: art_batch_rois,
            pv_roi_placeholder: pv_batch_rois,
            nc_patch_placeholder: nc_batch_patches,
            art_patch_placeholder: art_batch_patches,
            pv_patch_placeholder: pv_batch_patches,
            batch_size_placeholder: len(pv_batch_patches),
            attrs_placeholder: attrs
        }
        logits_v = sess.run(logits, feed_dict=feed_dict)
        logits_values.extend(np.argmax(logits_v, axis=1))
        if batch_count % 100 == 0:
            print logits_values
        batch_count += 1
        if epoch_end:
            break
    print logits_values
    return logits_values, labels


def generate_roi_feature_dataset(dataset, netname, model_path, feature_save_path, label_save_path,
                                 using_attribute_flag=True, using_clstm_flag=True, global_branch_flag=True,
                                 local_branch_flag=True, patch_size=5):
    '''
    对dataset下面的每一个slice生成测试的结果
    :param dataset: dataset的路径
    :return:
    '''
    nc_roi_placeholder = tf.placeholder(tf.float32,
                                        [None, config.ROI_IMAGE_HEIGHT, config.ROI_IMAGE_WIDTH, 3],
                                        name='nc_roi_placeholder')
    art_roi_placeholder = tf.placeholder(tf.float32,
                                         [None, config.ROI_IMAGE_HEIGHT, config.ROI_IMAGE_WIDTH, 3],
                                         name='art_roi_placeholder')
    pv_roi_placeholder = tf.placeholder(tf.float32,
                                        [None, config.ROI_IMAGE_HEIGHT, config.ROI_IMAGE_WIDTH, 3],
                                        name='pv_roi_placeholder')
    nc_patch_placeholder = tf.placeholder(tf.float32,
                                          [None, config.PATCH_IMAGE_HEIGHT, config.PATCH_IMAGE_WIDTH, 3],
                                          name='nc_patch_placeholder')
    art_patch_placeholder = tf.placeholder(tf.float32,
                                           [None, config.PATCH_IMAGE_HEIGHT, config.PATCH_IMAGE_WIDTH, 3],
                                           name='art_patch_placeholder')
    pv_patch_placeholder = tf.placeholder(tf.float32,
                                          [None, config.PATCH_IMAGE_HEIGHT, config.PATCH_IMAGE_WIDTH, 3],
                                          name='pv_patch_placeholder')
    batch_label_placeholder = tf.placeholder(tf.int32, [None, 1], name='batch_label_input')
    batch_attrs_placeholder = tf.placeholder(tf.float32, [None, 15], name='batch_attrs_input')
    batch_size_placeholder = tf.placeholder(tf.int32, [], name='batch_size')
    # label_placeholder = tf.placeholder(tf.int32, [None], name='label_placeholder')
    net = networks_with_attrs(nc_roi_placeholder, art_roi_placeholder, pv_roi_placeholder, nc_patch_placeholder,
                              art_patch_placeholder, pv_patch_placeholder, batch_attrs_placeholder, netname,
                              is_training=False, num_classes=config.num_classes, batch_size=batch_size_placeholder,
                              use_attribute_flag=using_attribute_flag, clstm_flag=using_clstm_flag,
                              global_branch_flag=global_branch_flag, local_branch_flag=local_branch_flag)
    logits = net.logits
    ce_loss, center_loss, gb_ce, lb_ce = net.build_loss(batch_label_placeholder, add_to_collection=False)
    predictions = []
    gpu_config = tf.ConfigProto()

    gpu_config.gpu_options.allow_growth = True
    roi_features = []
    with tf.Session(config=gpu_config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        # model_path = '/media/dl-box/HDD3/ld/PycharmProjects/GL_BD_LSTM/logs/model.ckpt-150809'
        saver = tf.train.Saver()
        print('restore from ', model_path)
        saver.restore(sess, model_path)
        batch_count = 0
        print(dataset)
        slice_names = os.listdir(dataset)
        print(slice_names)
        total_labels = []
        total_predictions = []
        roi_labels = []
        for idx, slice_name in enumerate(slice_names):
            if slice_name.startswith('.DS'):
                continue
            # if not slice_name.endswith('0'):
            #     continue
            print(slice_name, idx, ' / ', len(slice_names))
            cur_data_dir = os.path.join(dataset, slice_name)
            logits_values, sample_labels = generate_roi_feature_with_attributions(dataset, slice_name, patch_size,
                                                                                  sess, logits, nc_roi_placeholder,
                                                                                  art_roi_placeholder,
                                                                                  pv_roi_placeholder,
                                                                                  nc_patch_placeholder,
                                                                                  art_patch_placeholder,
                                                                                  pv_patch_placeholder,
                                                                                  batch_attrs_placeholder,
                                                                                  batch_size_placeholder)
            total_labels.extend(sample_labels)
            total_predictions.extend(logits_values)
            roi_feature = np.asarray([0., 0., 0., 0., 0.], np.float32)
            patch_num = len(logits_values) * 1.0
            for value in np.unique(logits_values):
                roi_feature[value] = np.sum(logits_values == value) * 1.0
            roi_feature /= patch_num
            roi_features.append(roi_feature)
            print(slice_name[-1], ' logits_values is ', logits_values)
            print('roi_feature is ', roi_feature)
            roi_labels.append(int(slice_name[-1]))
    print('the number of patches is ', len(total_labels))
    total_labels = np.asarray(total_labels, np.int32)
    total_predictions = np.asarray(total_predictions, np.int32)
    total_acc = np.sum(total_labels == total_predictions) / (1.0 * len(total_labels))
    print('the total acc is ', total_acc)
    for class_id in range(5):
        idx = np.where(total_labels == class_id)
        cur_predictions = total_predictions[idx]
        cur_gts = total_labels[idx]
        cur_acc = np.sum(cur_predictions == cur_gts) / (1.0 * len(cur_gts))
        print('the %d\'s acc is %.4f' % (class_id, cur_acc))
    roi_features = np.asarray(roi_features, np.float32)
    roi_labels = np.asarray(roi_labels, np.int32)
    np.save(feature_save_path, roi_features)
    np.save(label_save_path, roi_labels)


if __name__ == '__main__':
    restore_paras = {
        'model_path': '/media/dl-box/HDD3/ld/PycharmProjects/GL_BD_LSTM/logs/3x3/1/res50_original_decay_lr/model.ckpt-7672',
        'netname': 'res50',
        'stage_name': 'test',
        'dataset_dir': '/home/dl-box/ld/Documents/datasets/IEEEonMedicalImage_Splited/1',
        'roi_feature_save_dir': '/home/dl-box/ld/Documents/datasets/IEEEonMedicalImage_Splited/1/roi_feature/3x3/res50_original_decay_lr',
        'attribute_flag': True,
        'clstm_flag': True,
        'global_flag': True,
        'local_flag': True,
        'patch_size': 3,
        'gpu_id': '3'
    }
    # 0 9935
    # 1 9946
    os.environ['CUDA_VISIBLE_DEVICES'] = restore_paras['gpu_id']
    util.proc.set_proc_name('ld' + '_' + 'compute_' + restore_paras['stage_name'] + '_roi_feature_on' + '_GPU_'
                            + restore_paras['gpu_id'])
    if not os.path.exists(restore_paras['roi_feature_save_dir']):
        print('roi_feature_save_dir do not exists')
        os.mkdir(restore_paras['roi_feature_save_dir'])
        print('make it!')
    generate_roi_feature_dataset(
        os.path.join(restore_paras['dataset_dir'], restore_paras['stage_name']), restore_paras['netname'],
        restore_paras['model_path'],
        os.path.join(restore_paras['roi_feature_save_dir'],
                     restore_paras['netname'] + '_' + restore_paras['stage_name'] + '.npy'),
        os.path.join(restore_paras['roi_feature_save_dir'],
                     restore_paras['netname'] + '_' + restore_paras['stage_name'] + '_label' + '.npy'),
        using_attribute_flag=restore_paras['attribute_flag'], using_clstm_flag=restore_paras['clstm_flag'],
        global_branch_flag=restore_paras['global_flag'], local_branch_flag=restore_paras['local_flag'],
        patch_size=restore_paras['patch_size']
    )



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
from dataset.medicalImage import resolve_attribute_file
gpu_id = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id


def generate_paths(dataset_dir, stage_name, shuffling=True):
    cur_dataset_dir = os.path.join(dataset_dir, stage_name)
    slice_names = os.listdir(cur_dataset_dir)
    patch_size = 5
    RANDOM_SEED = 10
    nc_roi_paths = []
    art_roi_paths = []
    pv_roi_paths = []
    nc_patches_paths = []
    art_patches_paths = []
    pv_patches_paths = []
    labels = []
    for slice_name in slice_names:
        print slice_name
        cur_label = int(slice_name[-1])
        cur_slice_dir = os.path.join(cur_dataset_dir, slice_name)
        patches_dir = os.path.join(cur_slice_dir, '%dx%d' % (patch_size, patch_size))
        cur_nc_patches = sorted(glob(os.path.join(patches_dir, '*NC.jpg')))
        cur_art_patches = sorted(glob(os.path.join(patches_dir, '*ART.jpg')))
        cur_pv_pathes = sorted(glob(os.path.join(patches_dir, '*PV.jpg')))
        nc_roi_path = os.path.join(cur_slice_dir, 'NC.jpg')
        art_roi_path = os.path.join(cur_slice_dir, 'ART.jpg')
        pv_roi_path = os.path.join(cur_slice_dir, 'PV.jpg')

        nc_roi_paths.extend([nc_roi_path] * len(cur_nc_patches))
        art_roi_paths.extend([art_roi_path] * len(cur_art_patches))
        pv_roi_paths.extend([pv_roi_path] * len(cur_pv_pathes))

        nc_patches_paths.extend(cur_nc_patches)
        art_patches_paths.extend(cur_art_patches)
        pv_patches_paths.extend(cur_pv_pathes)
        labels.extend([cur_label] * len(cur_pv_pathes))
        if len(nc_roi_paths) != len(nc_patches_paths) or len(nc_roi_paths) != len(art_roi_paths) or len(
            nc_roi_paths) != len(art_patches_paths) or len(nc_roi_paths) != len(pv_patches_paths) or len(
            nc_roi_paths) != len(pv_roi_paths):
            print('the number is not equal')
            assert False

    nc_roi_paths = np.asarray(nc_roi_paths)
    art_roi_paths = np.asarray(art_roi_paths)
    pv_roi_paths = np.asarray(pv_roi_paths)
    nc_patches_paths = np.asarray(nc_patches_paths)
    art_patches_paths = np.asarray(art_patches_paths)
    pv_patches_paths = np.asarray(pv_patches_paths)
    labels = np.asarray(labels)
    if shuffling:
        random.seed(RANDOM_SEED)
        idx = range(len(nc_roi_paths))
        random.shuffle(idx)
        nc_roi_paths = nc_roi_paths[idx]
        nc_patches_paths = nc_patches_paths[idx]
        art_roi_paths = art_roi_paths[idx]
        art_patches_paths = art_patches_paths[idx]
        pv_roi_paths = pv_roi_paths[idx]
        pv_patches_paths = pv_patches_paths[idx]
        labels = labels[idx]
        print(len(nc_roi_paths), len(art_roi_paths), len(pv_roi_paths), len(nc_patches_paths), len(art_patches_paths),
              len(pv_patches_paths))
    return nc_roi_paths, art_roi_paths, pv_roi_paths, nc_patches_paths, art_patches_paths, pv_patches_paths, labels


def preprocessing_image(image, output_height, output_width):
    _R_MEAN = 164.0
    _G_MEAN = 164.0
    _B_MEAN = 164.0
    resized_image = cv2.resize(image, (output_height, output_width))
    resized_image = np.asarray(resized_image, np.float32)
    resized_image -= [_R_MEAN, _G_MEAN, _B_MEAN]
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


def generate_roi_feature(nc_path, art_path, pv_path, patch_size, sess, logits, nc_roi_placeholder, art_roi_placeholder,
                         pv_roi_placeholder, nc_patch_placeholder, art_patch_placeholder, pv_patch_placeholder,
                         batch_size_placeholder):
    nc_img = cv2.imread(nc_path)
    art_img = cv2.imread(art_path)
    pv_img = cv2.imread(pv_path)

    def _extract_patches(nc_img, art_img, pv_img, patch_size):
        nc_patches = []
        art_patches = []
        pv_patches = []
        width, height, _ = np.shape(nc_img)
        patch_step = 1
        if width * height >= 400:
            patch_step = int(math.sqrt(width * height / 100))
        for i in range(patch_size / 2, width - patch_size / 2, patch_step):
            for j in range(patch_size / 2, height - patch_size / 2, patch_step):
                cur_nc_patch = nc_img[i - patch_size / 2:i + patch_size / 2 + 1,
                            j - patch_size / 2: j + patch_size / 2 + 1, :]
                cur_art_patch = art_img[i - patch_size / 2:i + patch_size / 2 + 1,
                               j - patch_size / 2: j + patch_size / 2 + 1, :]
                cur_pv_patch = pv_img[i - patch_size / 2:i + patch_size / 2 + 1,
                               j - patch_size / 2: j + patch_size / 2 + 1, :]
                nc_patches.append(cur_nc_patch)
                art_patches.append(cur_art_patch)
                pv_patches.append(cur_pv_patch)
        return nc_patches, art_patches, pv_patches
    nc_patches, art_patches, pv_patches = _extract_patches(nc_img, art_img, pv_img, patch_size=patch_size)
    nc_rois = [nc_img] * len(nc_patches)
    art_rois = [art_img] * len(art_patches)
    pv_rois = [pv_img] * len(pv_patches)
    labels = [int(os.path.dirname(nc_path)[-1])] * len(pv_patches)
    dataset = Generate_Batch_Data_v2(nc_rois, art_rois, pv_rois, nc_patches, art_patches, pv_patches,
                                     config.val_batch_size, labels)
    generator = dataset.generate_next_batch()
    logits_values = []
    batch_count = 1
    while True:
        nc_batch_rois, art_batch_rois, pv_batch_rois, \
        nc_batch_patches, art_batch_patches, pv_batch_patches, \
        batch_label, epoch_end = generator.next()
        feed_dict = {
            nc_roi_placeholder: nc_batch_rois,
            art_roi_placeholder: art_batch_rois,
            pv_roi_placeholder: pv_batch_rois,
            nc_patch_placeholder: nc_batch_patches,
            art_patch_placeholder: art_batch_patches,
            pv_patch_placeholder: pv_batch_patches,
            batch_size_placeholder: len(pv_batch_patches)
        }
        logits_v = sess.run(logits, feed_dict=feed_dict)
        logits_values.extend(np.argmax(logits_v, axis=1))
        if batch_count % 100 == 0:
            print logits_values
        batch_count += 1
        if epoch_end:
            break
    print logits_values
    return logits_values


def generate_roi_feature_with_attributions(nc_path, art_path, pv_path, patch_size, sess, logits, nc_roi_placeholder, art_roi_placeholder,
                         pv_roi_placeholder, nc_patch_placeholder, art_patch_placeholder, pv_patch_placeholder, attrs_placeholder,
                         batch_size_placeholder):
    nc_img = cv2.imread(nc_path)
    art_img = cv2.imread(art_path)
    pv_img = cv2.imread(pv_path)
    nc_attrs_path = os.path.join(os.path.dirname(nc_path), 'NC_attributes.txt')
    art_attrs_path = os.path.join(os.path.dirname(nc_path), 'ART_attributes.txt')
    pv_attrs_path = os.path.join(os.path.dirname(nc_path), 'PV_attributes.txt')
    nc_attr = resolve_attribute_file(nc_attrs_path)
    art_attr = resolve_attribute_file(art_attrs_path)
    pv_attr = resolve_attribute_file(pv_attrs_path)

    def _extract_patches(nc_img, art_img, pv_img, patch_size):
        nc_patches = []
        art_patches = []
        pv_patches = []
        width, height, _ = np.shape(nc_img)
        patch_step = 1
        if width * height >= 400:
            patch_step = int(math.sqrt(width * height / 100))
        for i in range(patch_size / 2, width - patch_size / 2, patch_step):
            for j in range(patch_size / 2, height - patch_size / 2, patch_step):
                cur_nc_patch = nc_img[i - patch_size / 2:i + patch_size / 2 + 1,
                            j - patch_size / 2: j + patch_size / 2 + 1, :]
                cur_art_patch = art_img[i - patch_size / 2:i + patch_size / 2 + 1,
                               j - patch_size / 2: j + patch_size / 2 + 1, :]
                cur_pv_patch = pv_img[i - patch_size / 2:i + patch_size / 2 + 1,
                               j - patch_size / 2: j + patch_size / 2 + 1, :]
                nc_patches.append(cur_nc_patch)
                art_patches.append(cur_art_patch)
                pv_patches.append(cur_pv_patch)
        return nc_patches, art_patches, pv_patches
    nc_patches, art_patches, pv_patches = _extract_patches(nc_img, art_img, pv_img, patch_size=patch_size)
    nc_rois = [nc_img] * len(nc_patches)
    art_rois = [art_img] * len(art_patches)
    pv_rois = [pv_img] * len(pv_patches)
    nc_attrs = [nc_attr] * len(nc_patches)
    art_attrs = [art_attr] * len(art_patches)
    pv_attrs = [pv_attr] * len(pv_patches)


    labels = [int(os.path.dirname(nc_path)[-1])] * len(pv_patches)
    # dataset = Generate_Batch_Data_v2(nc_rois, art_rois, pv_rois, nc_patches, art_patches, pv_patches,
    #                                  config.val_batch_size, labels)
    dataset = Generate_Batch_Data_v2_with_attributes(nc_rois, art_rois, pv_rois, nc_patches, art_patches,
                                                     pv_patches, nc_attrs, art_attrs, pv_attrs, config.val_batch_size,
                                                     labels)
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
    return logits_values


def generate_roi_feature_dataset(dataset, netname, model_path, feature_save_path, attribute_flag=False):
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
                                          [None, config.ROI_IMAGE_HEIGHT, config.ROI_IMAGE_WIDTH, 3],
                                          name='nc_patch_placeholder')
    art_patch_placeholder = tf.placeholder(tf.float32,
                                           [None, config.ROI_IMAGE_HEIGHT, config.ROI_IMAGE_WIDTH, 3],
                                           name='art_patch_placeholder')
    pv_patch_placeholder = tf.placeholder(tf.float32,
                                          [None, config.ROI_IMAGE_HEIGHT, config.ROI_IMAGE_WIDTH, 3],
                                          name='pv_patch_placeholder')
    batch_label_placeholder = tf.placeholder(tf.int32, [None, 1], name='batch_label_input')
    if attribute_flag:
        batch_attrs_placeholder = tf.placeholder(tf.float32, [None, 15], name='batch_attrs_input')
    batch_size_placeholder = tf.placeholder(tf.int32, [], name='batch_size')
    # label_placeholder = tf.placeholder(tf.int32, [None], name='label_placeholder')
    if attribute_flag:
        net = networks_with_attrs(nc_roi_placeholder, art_roi_placeholder, pv_roi_placeholder, nc_patch_placeholder,
                                  art_patch_placeholder, pv_patch_placeholder, batch_attrs_placeholder, netname,
                                  is_training=False, num_classes=config.num_classes, batch_size=batch_size_placeholder)
    else:
        net = networks(nc_roi_placeholder, art_roi_placeholder, pv_roi_placeholder, nc_patch_placeholder,
                       art_patch_placeholder, pv_patch_placeholder, base_name=netname, is_training=False,
                       num_classes=config.num_classes, batch_size=batch_size_placeholder)
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

        slice_names = os.listdir(dataset)
        for slice_name in slice_names:
            # if not slice_name.endswith('0'):
            #     continue
            print(slice_name)
            cur_data_dir = os.path.join(dataset, slice_name)
            nc_path = os.path.join(cur_data_dir, 'NC.jpg')
            art_path = os.path.join(cur_data_dir, 'ART.jpg')
            pv_path = os.path.join(cur_data_dir, 'PV.jpg')
            if attribute_flag:
                logits_values = generate_roi_feature_with_attributions(nc_path, art_path, pv_path, config.patch_size,
                                                                       sess, logits, nc_roi_placeholder,
                                                                       art_roi_placeholder, pv_roi_placeholder,
                                                                       nc_patch_placeholder, art_patch_placeholder,
                                                                       pv_patch_placeholder, batch_attrs_placeholder,
                                                                       batch_size_placeholder)
            else:
                logits_values = generate_roi_feature(nc_path, art_path, pv_path, config.patch_size, sess, logits,
                                                     nc_roi_placeholder, art_roi_placeholder, pv_roi_placeholder,
                                                     nc_patch_placeholder, art_patch_placeholder, pv_patch_placeholder,
                                                     batch_size_placeholder)
            roi_feature = np.asarray([0., 0., 0., 0., 0.], np.float32)
            patch_num = len(logits_values) * 1.0
            for value in np.unique(logits_values):
                roi_feature[value] = np.sum(logits_values == value) * 1.0
            roi_feature /= patch_num
            roi_features.append(roi_feature)
            print(nc_path[-1], ' logits_values is ', logits_values)
            print('roi_feature is ', roi_feature)
    roi_features = np.asarray(roi_features, np.float32)
    np.save(feature_save_path, roi_features)


if __name__ == '__main__':
    util.proc.set_proc_name('ld_test_on' + '_' + 'medical_image_classification' + '_GPU_' + gpu_id)
    restore_paras = {
        'model_path': '/media/dl-box/HDD3/ld/PycharmProjects/GL_BD_LSTM/logs/res50_attribute_val_gl/model.ckpt-10780',
        'netname': 'res50',
        'stage_name': 'train',
        'dataset_dir': '/home/dl-box/ld/Documents/datasets/IEEEonMedicalImage_Splited/JPG/0_attribute',
        'roi_feature_save_dir': '/home/dl-box/ld/Documents/datasets/IEEEonMedicalImage_Splited/JPG/0_attribute',
        'attribute_flag': True
    }

    generate_roi_feature_dataset(
        os.path.join(restore_paras['dataset_dir'], restore_paras['stage_name']), restore_paras['netname'],
        restore_paras['model_path'],
        os.path.join(restore_paras['roi_feature_save_dir'],
                     restore_paras['netname'] + '_' +restore_paras['stage_name'] + '.npy'),
        restore_paras['attribute_flag']
    )



# -*- coding=utf-8 -*-
import tensorflow as tf
from models.research.slim.preprocessing.vgg_preprocessing import preprocess_image_GL
import config
from networks import networks, networks_with_attrs
import os
import numpy as np
from glob import glob
import cv2
import random
from dataset.medicalImage import resolve_attribute_file, read_mhd_image, get_liver_path, get_filled_mask, read_liver_mask, compute_mean_img
from dataset.convert2jpg import calculate_mask_attributes, extract_patches
from config import RANDOM_SEED

os.environ['CUDA_VISIBLE_DEVICES']='2'


def generate_patches_with_attributions(dataset_dir, stage_name, shuffling=True):
    cur_dataset_dir = os.path.join(dataset_dir, stage_name)
    slice_names = os.listdir(cur_dataset_dir)
    patch_size = 5
    nc_rois = []
    art_rois = []
    pv_rois = []
    nc_patches = []
    art_patches = []
    pv_patches = []
    labels = []
    nc_attrs = []
    pv_attrs = []
    art_attrs = []

    for slice_name in slice_names:
        if slice_name.startswith('.DS'):
            continue
        print slice_name
        cur_label = int(slice_name[-1])
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
        print('nc mean is ', nc_roi_final.mean((0, 1)))
        print('art mean is ', art_roi_final.mean((0, 1)))
        print('pv mean is ', pv_roi_final.mean((0, 1)))
        cur_nc_patches, cur_art_patches, cur_pv_patches = extract_patches(nc_roi_final, art_roi_final, pv_roi_final,
                                                                          patch_size=patch_size)
        nc_patches.extend(cur_nc_patches)
        art_patches.extend(cur_art_patches)
        pv_patches.extend(cur_pv_patches)
        labels.extend([cur_label] * len(cur_pv_patches))
        nc_rois.extend([nc_roi_final] * len(cur_nc_patches))
        art_rois.extend([art_roi_final] * len(cur_art_patches))
        pv_rois.extend([pv_roi_final] * len(cur_pv_patches))
        nc_attrs.extend([nc_attr] * len(cur_nc_patches))
        art_attrs.extend([art_attr] * len(cur_art_patches))
        pv_attrs.extend([pv_attr] * len(cur_pv_patches))

        if len(nc_rois) != len(nc_patches) or len(nc_rois) != len(art_patches) or len(
                nc_patches) != len(art_patches) or len(nc_rois) != len(pv_patches) or len(
            nc_rois) != len(pv_rois):
            print('the number is not equal')
            assert False

    nc_rois = np.asarray(nc_rois)
    art_rois = np.asarray(art_rois)
    pv_rois = np.asarray(pv_rois)
    nc_patches = np.asarray(nc_patches)
    art_patches = np.asarray(art_patches)
    pv_patches = np.asarray(pv_patches)
    nc_attrs = np.asarray(nc_attrs)
    art_attrs = np.asarray(art_attrs)
    pv_attrs = np.asarray(pv_attrs)
    labels = np.asarray(labels)
    nc_attrs /= config.MAX_ATTRS
    art_attrs /= config.MAX_ATTRS
    pv_attrs /= config.MAX_ATTRS
    if shuffling:
        random.seed(RANDOM_SEED)
        idx = range(len(nc_rois))
        random.shuffle(idx)
        nc_rois = nc_rois[idx]
        art_rois = art_rois[idx]
        pv_rois = pv_rois[idx]
        nc_patches = nc_patches[idx]
        art_patches = art_patches[idx]
        pv_patches = pv_patches[idx]

        nc_attrs = nc_attrs[idx]
        art_attrs = art_attrs[idx]
        pv_attrs = pv_attrs[idx]
        labels = labels[idx]
        print(len(nc_rois), len(art_rois), len(pv_rois), len(nc_patches), len(art_patches),
              len(pv_patches), len(nc_attrs), len(art_attrs), len(pv_attrs), len(labels))
    attrs = np.concatenate([nc_attrs, art_attrs, pv_attrs], axis=1)
    print('the shape of nc_roi is ', np.shape(nc_rois))
    print('nc_roi mean is ', compute_mean_img(nc_rois))
    print('art_roi mean is ', compute_mean_img(art_rois))
    print('pv_roi mean is ', compute_mean_img(pv_rois))
    print('attrs shape is ', np.shape(attrs))
    return nc_rois, art_rois, pv_rois, nc_patches, art_patches, pv_patches, attrs, labels


class Generate_Batch_Data_with_attributions:
    def __init__(self, nc_rois, art_rois, pv_rois, nc_patches, art_patches, pv_patches, attrs, batch_size, labels=None):
        self.nc_rois = nc_rois
        self.art_rois = art_rois
        self.pv_rois = pv_rois
        self.nc_patches = nc_patches
        self.art_patches = art_patches
        self.pv_patches = pv_patches
        self.batch_size = batch_size
        self.attrs = attrs
        self.labels = labels
        self.start_index = 0

    def preprocessing_image(self, image, output_height, output_width):
        _R_MEAN = 164.0
        _G_MEAN = 164.0
        _B_MEAN = 164.0
        resized_image = cv2.resize(image, (output_height, output_width))
        resized_image = np.asarray(resized_image, np.float32)
        # resized_image -= [_R_MEAN, _G_MEAN, _B_MEAN]
        return resized_image

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
            batch_attrs = self.attrs[self.start_index: end_index]
            # nc_batch_rois = [cv2.imread(nc_batch_roi_path) for nc_batch_roi_path in nc_batch_roi_paths]
            # art_batch_rois = [cv2.imread(art_batch_roi_path) for art_batch_roi_path in art_batch_roi_paths]
            # pv_batch_rois = [cv2.imread(pv_batch_roi_path) for pv_batch_roi_path in pv_batch_roi_paths]
            # art_batch_patches = [cv2.imread(art_patch_path) for art_patch_path in art_batch_patch_paths]
            # nc_batch_patches = [cv2.imread(nc_patch_path) for nc_patch_path in nc_batch_patch_paths]
            # pv_batch_patches = [cv2.imread(pv_patch_path) for pv_patch_path in pv_batch_patch_paths]
            nc_batch_rois_preprocessed = [
                self.preprocessing_image(nc_roi, config.ROI_IMAGE_HEIGHT, config.ROI_IMAGE_WIDTH) for nc_roi in
                nc_batch_rois]
            art_batch_rois_preprocessed = [
                self.preprocessing_image(art_roi, config.ROI_IMAGE_HEIGHT, config.ROI_IMAGE_WIDTH) for art_roi in
                art_batch_rois]
            pv_batch_rois_preprocessed = [
                self.preprocessing_image(pv_roi, config.ROI_IMAGE_HEIGHT, config.ROI_IMAGE_WIDTH) for pv_roi in
                pv_batch_rois]

            nc_batch_patches_preprocessed = [
                self.preprocessing_image(nc_patch, config.ROI_IMAGE_HEIGHT, config.ROI_IMAGE_WIDTH) for nc_patch in
                nc_batch_patches
            ]
            art_batch_patches_preprocessed = [
                self.preprocessing_image(art_patch, config.ROI_IMAGE_HEIGHT, config.ROI_IMAGE_WIDTH) for art_patch in
                art_batch_patches
            ]
            pv_batch_patches_preprocessed = [
                self.preprocessing_image(pv_patch, config.ROI_IMAGE_HEIGHT, config.ROI_IMAGE_WIDTH) for pv_patch in
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
                      batch_attrs, batch_labels, epoch_end
            else:
                yield nc_batch_rois_preprocessed, art_batch_rois_preprocessed, pv_batch_rois_preprocessed, \
                      nc_batch_patches_preprocessed, art_batch_patches_preprocessed, pv_batch_patches_preprocessed, \
                      batch_attrs, epoch_end


def evulate_imgs_batch_with_attributions(nc_rois, art_rois, pv_rois, nc_patches, art_patches, pv_patches, attrs,
                                         labels, netname, model_path):
    batch_dataset = Generate_Batch_Data_with_attributions(nc_rois, art_rois, pv_rois, nc_patches, art_patches,
                                                          pv_patches, attrs, config.val_batch_size, labels)
    generator = batch_dataset.generate_next_batch()
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
    batch_size_placeholder = tf.placeholder(tf.int32, [], name='batch_size')
    attrs_placeholder = tf.placeholder(tf.float32, [None, 15], 'attributions_placeholder')
    # label_placeholder = tf.placeholder(tf.int32, [None], name='label_placeholder')

    net = networks_with_attrs(nc_roi_placeholder, art_roi_placeholder, pv_roi_placeholder, nc_patch_placeholder,
                              art_patch_placeholder, pv_patch_placeholder, attrs_placeholder, base_name=netname,
                              is_training=True, num_classes=config.num_classes, batch_size=batch_size_placeholder)
    logits = net.logits
    ce_loss, center_loss, gb_ce, lb_ce = net.build_loss(batch_label_placeholder, add_to_collection=False)
    predictions = []
    gts = labels
    gpu_config = tf.ConfigProto()

    gpu_config.gpu_options.allow_growth = True
    with tf.Session(config=gpu_config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        # model_path = '/media/dl-box/HDD3/ld/PycharmProjects/GL_BD_LSTM/logs/model.ckpt-150809'
        saver = tf.train.Saver()
        print('restore from ', model_path)
        saver.restore(sess, model_path)
        batch_count = 0
        ce_loss_values = []
        center_loss_values = []

        while True:
            nc_batch_rois, art_batch_rois, pv_batch_rois, \
            nc_batch_patches, art_batch_patches, pv_batch_patches, \
            batch_attrs, batch_label, epoch_end = generator.next()
            # print('batch_label is ', batch_label, np.shape(batch_label))
            logits_v, ce_loss_v = sess.run([logits, ce_loss], feed_dict={
                nc_roi_placeholder: nc_batch_rois,
                art_roi_placeholder: art_batch_rois,
                pv_roi_placeholder: pv_batch_rois,
                nc_patch_placeholder: nc_batch_patches,
                art_patch_placeholder: art_batch_patches,
                pv_patch_placeholder: pv_batch_patches,
                batch_size_placeholder: len(pv_batch_patches),
                attrs_placeholder: batch_attrs,
                batch_label_placeholder: np.expand_dims(batch_label, axis=1)
            })

            ce_loss_values.append(ce_loss_v)
            print(np.mean(ce_loss_values))
            predictions.extend(np.argmax(logits_v, axis=1))
            # print(np.argmax(logits_v, axis=1), batch_label)
            if batch_count % 100 == 0 and batch_count != 0:
                print('%d/%d\n' % (batch_count * config.val_batch_size, len(nc_rois)))
            batch_count += 1
            if epoch_end:
                break
    gts = np.asarray(gts, np.uint8)
    predictions = np.asarray(predictions, np.uint8)
    total_acc = np.sum(gts == predictions) / (1.0 * len(gts))
    print('the total acc is ', total_acc)
    for class_id in range(5):
        idx = np.where(gts == class_id)
        cur_predictions = predictions[idx]
        cur_gts = gts[idx]
        cur_acc = np.sum(cur_predictions == cur_gts) / (1.0 * len(cur_gts))
        print('the %d\'s acc is %.4f' % (class_id, cur_acc))


if __name__ == '__main__':
    restore_paras = {
        'model_path': '/media/dl-box/HDD3/ld/PycharmProjects/GL_BD_LSTM/logs/res50_original/model.ckpt-3059',
        'netname': 'res50',
        'stage_name': 'val',
        'dataset_dir': '/home/dl-box/ld/Documents/datasets/IEEEonMedicalImage_Splited/0'
    }
    attribution_flag = True
    if attribution_flag:
        nc_rois, art_rois, pv_rois, nc_patches, art_patches, pv_patches, attrs, labels = \
            generate_patches_with_attributions(
                restore_paras['dataset_dir'],
                restore_paras['stage_name'],
                True)

        evulate_imgs_batch_with_attributions(
            nc_rois, art_rois, pv_rois, nc_patches, art_patches, pv_patches, attrs,
            labels, model_path=restore_paras['model_path'], netname=restore_paras['netname']
        )


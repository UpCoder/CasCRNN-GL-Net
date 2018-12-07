import tensorflow as tf
from models.research.slim.preprocessing.vgg_preprocessing import preprocess_image_GL
import config
from networks import networks, networks_with_attrs
import os
import numpy as np
from glob import glob
import cv2
import random
from dataset.medicalImage import resolve_attribute_file

os.environ['CUDA_VISIBLE_DEVICES']='2'


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


def generate_paths_with_attributions(dataset_dir, stage_name, shuffling=True):
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
    attrs = []
    for slice_name in slice_names:
        print slice_name
        cur_label = int(slice_name[-1])
        cur_slice_dir = os.path.join(cur_dataset_dir, slice_name)
        patches_dir = os.path.join(cur_slice_dir, '%dx%d' % (patch_size, patch_size))
        cur_nc_patches = sorted(glob(os.path.join(patches_dir, '*NC.jpg')))
        cur_art_patches = sorted(glob(os.path.join(patches_dir, '*ART.jpg')))
        cur_pv_patches = sorted(glob(os.path.join(patches_dir, '*PV.jpg')))
        nc_roi_path = os.path.join(cur_slice_dir, 'NC.jpg')
        art_roi_path = os.path.join(cur_slice_dir, 'ART.jpg')
        pv_roi_path = os.path.join(cur_slice_dir, 'PV.jpg')

        nc_attrs_path = os.path.join(cur_slice_dir, 'NC_attributes.txt')
        art_attrs_path = os.path.join(cur_slice_dir, 'ART_attributes.txt')
        pv_attrs_path = os.path.join(cur_slice_dir, 'PV_attributes.txt')

        nc_attrs = np.asarray(resolve_attribute_file(nc_attrs_path), np.float32)
        art_attrs = np.asarray(resolve_attribute_file(art_attrs_path), np.float32)
        pv_attrs = np.asarray(resolve_attribute_file(pv_attrs_path), np.float32)
        # nc_attrs -= config.MEAN_ATTRS
        # art_attrs -= config.MEAN_ATTRS
        # pv_attrs -= config.MEAN_ATTRS
        nc_attrs /= config.MAX_ATTRS
        art_attrs /= config.MAX_ATTRS
        pv_attrs /= config.MAX_ATTRS
        cur_attrs = np.concatenate([nc_attrs, art_attrs, pv_attrs], axis=-1)

        nc_roi_paths.extend([nc_roi_path] * len(cur_nc_patches))
        art_roi_paths.extend([art_roi_path] * len(cur_art_patches))
        pv_roi_paths.extend([pv_roi_path] * len(cur_pv_patches))
        attrs.extend([cur_attrs] * len(cur_pv_patches))

        nc_patches_paths.extend(cur_nc_patches)
        art_patches_paths.extend(cur_art_patches)
        pv_patches_paths.extend(cur_pv_patches)
        labels.extend([cur_label] * len(cur_pv_patches))
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
    attrs = np.asarray(attrs)
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
        attrs = attrs[idx]
        print(len(nc_roi_paths), len(art_roi_paths), len(pv_roi_paths), len(nc_patches_paths), len(art_patches_paths),
              len(pv_patches_paths))
    return nc_roi_paths, art_roi_paths, pv_roi_paths, nc_patches_paths, art_patches_paths, pv_patches_paths, attrs, \
           labels


class Generate_Batch_Data_with_attributions:
    def __init__(self, nc_rois_paths, art_rois_paths, pv_rois_paths, nc_patches_paths, art_patches_paths, pv_patches_paths, attrs, batch_size, labels=None):
        self.nc_roi_paths = nc_rois_paths
        self.art_roi_paths = art_rois_paths
        self.pv_roi_paths = pv_rois_paths
        self.nc_patch_paths = nc_patches_paths
        self.art_patch_paths = art_patches_paths
        self.pv_patch_paths = pv_patches_paths
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
        resized_image -= [_R_MEAN, _G_MEAN, _B_MEAN]
        return resized_image

    def generate_next_batch(self):
        while self.start_index < len(self.nc_roi_paths):
            epoch_end = False
            end_index = self.start_index + self.batch_size
            if end_index > len(nc_roi_paths):
                end_index = len(nc_roi_paths)
            nc_batch_roi_paths = self.nc_roi_paths[self.start_index: end_index]
            art_batch_roi_paths = self.art_roi_paths[self.start_index: end_index]
            pv_batch_roi_paths = self.pv_roi_paths[self.start_index: end_index]
            nc_batch_patch_paths = self.nc_patch_paths[self.start_index: end_index]
            art_batch_patch_paths = self.art_patch_paths[self.start_index: end_index]
            pv_batch_patch_paths = self.pv_patch_paths[self.start_index: end_index]
            batch_attrs = self.attrs[self.start_index: end_index]
            nc_batch_rois = [cv2.imread(nc_batch_roi_path) for nc_batch_roi_path in nc_batch_roi_paths]
            art_batch_rois = [cv2.imread(art_batch_roi_path) for art_batch_roi_path in art_batch_roi_paths]
            pv_batch_rois = [cv2.imread(pv_batch_roi_path) for pv_batch_roi_path in pv_batch_roi_paths]
            art_batch_patches = [cv2.imread(art_patch_path) for art_patch_path in art_batch_patch_paths]
            nc_batch_patches = [cv2.imread(nc_patch_path) for nc_patch_path in nc_batch_patch_paths]
            pv_batch_patches = [cv2.imread(pv_patch_path) for pv_patch_path in pv_batch_patch_paths]
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
            if end_index == len(nc_roi_paths):
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


class Generate_Batch_Data:
    def __init__(self, nc_rois_paths, art_rois_paths, pv_rois_paths, nc_patches_paths, art_patches_paths, pv_patches_paths, batch_size, labels=None):
        self.nc_roi_paths = nc_rois_paths
        self.art_roi_paths = art_rois_paths
        self.pv_roi_paths = pv_rois_paths
        self.nc_patch_paths = nc_patches_paths
        self.art_patch_paths = art_patches_paths
        self.pv_patch_paths = pv_patches_paths
        self.batch_size = batch_size
        self.labels = labels
        self.start_index = 0

    def preprocessing_image(self, image, output_height, output_width):
        _R_MEAN = 164.0
        _G_MEAN = 164.0
        _B_MEAN = 164.0
        resized_image = cv2.resize(image, (output_height, output_width))
        resized_image = np.asarray(resized_image, np.float32)
        resized_image -= [_R_MEAN, _G_MEAN, _B_MEAN]
        return resized_image

    def generate_next_batch(self):
        while self.start_index < len(self.nc_roi_paths):
            epoch_end = False
            end_index = self.start_index + self.batch_size
            if end_index > len(nc_roi_paths):
                end_index = len(nc_roi_paths)
            nc_batch_roi_paths = self.nc_roi_paths[self.start_index: end_index]
            art_batch_roi_paths = self.art_roi_paths[self.start_index: end_index]
            pv_batch_roi_paths = self.pv_roi_paths[self.start_index: end_index]
            nc_batch_patch_paths = self.nc_patch_paths[self.start_index: end_index]
            art_batch_patch_paths = self.art_patch_paths[self.start_index: end_index]
            pv_batch_patch_paths = self.pv_patch_paths[self.start_index: end_index]
            nc_batch_rois = [cv2.imread(nc_batch_roi_path) for nc_batch_roi_path in nc_batch_roi_paths]
            art_batch_rois = [cv2.imread(art_batch_roi_path) for art_batch_roi_path in art_batch_roi_paths]
            pv_batch_rois = [cv2.imread(pv_batch_roi_path) for pv_batch_roi_path in pv_batch_roi_paths]
            art_batch_patches = [cv2.imread(art_patch_path) for art_patch_path in art_batch_patch_paths]
            nc_batch_patches = [cv2.imread(nc_patch_path) for nc_patch_path in nc_batch_patch_paths]
            pv_batch_patches = [cv2.imread(pv_patch_path) for pv_patch_path in pv_batch_patch_paths]
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
            if end_index == len(nc_roi_paths):
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


def evulate_imgs_batch_with_attributions(nc_rois_paths, art_rois_paths, pv_rois_paths, nc_patches_paths, art_patches_paths,
                                         pv_patches_paths, attrs, labels, netname, model_path):
    batch_dataset = Generate_Batch_Data_with_attributions(nc_rois_paths, art_rois_paths, pv_rois_paths,
                                                          nc_patches_paths, art_patches_paths, pv_patches_paths, attrs,
                                                          config.val_batch_size, labels)
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
                print('%d/%d\n' % (batch_count * config.val_batch_size, len(nc_rois_paths)))
            batch_count += 1
            if epoch_end:
                break
    # with open('./evulation.txt', 'w') as save_txt_file:
    #     lines = []
    #     for gt, prediction in zip(gts, predictions):
    #         line = '%d %d\n' % (prediction, gt)
    #         lines.append(line)
    #     save_txt_file.writelines(lines)
    #     save_txt_file.close()

    # calculate the accuracy
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


def evulate_imgs_batch(nc_rois_paths, art_rois_paths, pv_rois_paths, nc_patches_paths, art_patches_paths,
                       pv_patches_paths, labels, netname, model_path):
    batch_dataset = Generate_Batch_Data(nc_rois_paths, art_rois_paths, pv_rois_paths, nc_patches_paths,
                                        art_patches_paths, pv_patches_paths, config.val_batch_size, labels)
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
    # label_placeholder = tf.placeholder(tf.int32, [None], name='label_placeholder')

    net = networks(nc_roi_placeholder, art_roi_placeholder, pv_roi_placeholder, nc_patch_placeholder,
                   art_patch_placeholder, pv_patch_placeholder, base_name=netname, is_training=False,
                   num_classes=config.num_classes, batch_size=batch_size_placeholder)
    logits = net.logits
    ce_loss, center_loss, gb_ce, lb_ce = net.build_loss(batch_label_placeholder)
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
            batch_label, epoch_end = generator.next()
            # print('batch_label is ', batch_label, np.shape(batch_label))
            logits_v, ce_loss_v = sess.run([logits, ce_loss], feed_dict={
                nc_roi_placeholder: nc_batch_rois,
                art_roi_placeholder: art_batch_rois,
                pv_roi_placeholder: pv_batch_rois,
                nc_patch_placeholder: nc_batch_patches,
                art_patch_placeholder: art_batch_patches,
                pv_patch_placeholder: pv_batch_patches,
                batch_size_placeholder: len(pv_batch_patches),
                batch_label_placeholder: np.expand_dims(batch_label, axis=1)
            })
            ce_loss_values.append(ce_loss_v)
            print(np.mean(ce_loss_values))
            predictions.extend(np.argmax(logits_v, axis=1))
            if batch_count % 100 == 0 and batch_count != 0:
                print('%d/%d\n' % (batch_count * config.val_batch_size, len(nc_rois_paths)))
            batch_count += 1
            if epoch_end:
                break
    # with open('./evulation.txt', 'w') as save_txt_file:
    #     lines = []
    #     for gt, prediction in zip(gts, predictions):
    #         line = '%d %d\n' % (prediction, gt)
    #         lines.append(line)
    #     save_txt_file.writelines(lines)
    #     save_txt_file.close()

    # calculate the accuracy
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


def evulate_imgs(nc_rois_paths, art_rois_paths, pv_rois_paths, nc_patches_paths, art_patches_paths,
                 pv_patches_paths, labels, load_result=False, netname='res50', model_path=None):
    if load_result:
        with open('./evulation.txt', 'r') as read_txt_file:
            lines = read_txt_file.readlines()
            res = [[splited_line[0], splited_line[1]] for splited_line in [line.split(' ') for line in lines]]
            res = np.asarray(res, np.uint8)
            predictions = res[:, 0]
            gts = res[:, 1]
            print np.shape(res)
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
            return

    nc_roi_placeholder = tf.placeholder(tf.uint8, [None, None, 3], name='nc_roi_placeholder')
    art_roi_placeholder = tf.placeholder(tf.uint8, [None, None, 3], name='art_roi_placeholder')
    pv_roi_placeholder = tf.placeholder(tf.uint8, [None, None, 3], name='pv_roi_placeholder')
    nc_patch_placeholder = tf.placeholder(tf.uint8, [None, None, 3], name='nc_patch_placeholder')
    art_patch_placeholder = tf.placeholder(tf.uint8, [None, None, 3], name='art_patch_placeholder')
    pv_patch_placeholder = tf.placeholder(tf.uint8, [None, None, 3], name='pv_patch_placeholder')
    label_placeholder = tf.placeholder(tf.int32, [None], name='label_placeholder')
    nc_roi, art_roi, pv_roi, nc_patch, art_patch, pv_patch = preprocess_image_GL(nc_roi_placeholder,
                                                                                 art_roi_placeholder,
                                                                                 pv_roi_placeholder,
                                                                                 nc_patch_placeholder,
                                                                                 art_patch_placeholder,
                                                                                 pv_patch_placeholder,
                                                                                 config.ROI_IMAGE_HEIGHT,
                                                                                 config.ROI_IMAGE_HEIGHT,
                                                                                 is_training=False)

    nc_roi = tf.expand_dims(nc_roi, axis=0)
    art_roi = tf.expand_dims(art_roi, axis=0)
    pv_roi = tf.expand_dims(pv_roi, axis=0)
    nc_patch = tf.expand_dims(nc_patch, axis=0)
    art_patch = tf.expand_dims(art_patch, axis=0)
    pv_patch = tf.expand_dims(pv_patch, axis=0)
    label = tf.expand_dims(label_placeholder, axis=1)
    net = networks(nc_roi, art_roi, pv_roi, nc_patch, art_patch, pv_patch, base_name=netname, is_training=False,
                   num_classes=config.num_classes, batch_size=1)
    logits = net.logits
    ce_loss, center_loss, gb_ce, lb_ce = net.build_loss(label)
    predictions = []
    gts = []
    gpu_config = tf.ConfigProto()

    gpu_config.gpu_options.allow_growth = True
    with tf.Session(config=gpu_config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        # model_path = '/media/dl-box/HDD3/ld/PycharmProjects/GL_BD_LSTM/logs/vgg16/model.ckpt-28218'
        saver = tf.train.Saver()
        print('restore from ', model_path)
        saver.restore(sess, model_path)
        ce_loss_values = []
        center_loss_values = []
        for idx, (
        nc_roi_path, art_roi_path, pv_roi_path, nc_patch_path, art_patch_path, pv_patch_path, label) in enumerate(
            zip(nc_rois_paths, art_rois_paths, pv_rois_paths, nc_patches_paths, art_patches_paths, pv_patches_paths,
                labels)):
            logits_v, ce_loss_v, center_loss_v = sess.run([logits, ce_loss, center_loss], feed_dict={
                nc_roi_placeholder: cv2.imread(nc_roi_path),
                art_roi_placeholder: cv2.imread(art_roi_path),
                pv_roi_placeholder: cv2.imread(pv_roi_path),
                nc_patch_placeholder: cv2.imread(nc_patch_path),
                art_patch_placeholder: cv2.imread(art_patch_path),
                pv_patch_placeholder: cv2.imread(pv_patch_path),
                label_placeholder: [label]
            })

            ce_loss_values.append(ce_loss_v)
            center_loss_values.append(center_loss_v)
            # print(label, np.argmax(logits_v))
            print(np.mean(ce_loss_values), np.mean(center_loss_values))
            gts.append(label)
            predictions.append(np.argmax(logits_v))
            # print np.argmax(logits_v), label
            if idx % 100 == 0 and idx != 0:
                print('%d/%d\n' % (idx, len(nc_rois_paths)))

    with open('./evulation.txt', 'w') as save_txt_file:
        lines = []
        for gt, prediction in zip(gts, predictions):
            line = '%d %d\n' % (prediction, gt)
            lines.append(line)
        save_txt_file.writelines(lines)
        save_txt_file.close()

    # calculate the accuracy
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
        'model_path': '/media/dl-box/HDD3/ld/PycharmProjects/GL_BD_LSTM/logs/res50_attribute_val_gl/model.ckpt-10780',
        'netname': 'res50',
        'stage_name': 'val',
    }
    attribution_flag = True
    if attribution_flag:
        nc_roi_paths, art_roi_paths, pv_roi_paths, nc_patches_paths, art_patches_paths, pv_patches_paths, attrs, labels = \
            generate_paths_with_attributions(
                                            '/home/dl-box/ld/Documents/datasets/IEEEonMedicalImage_Splited/JPG/0_attribute',
                                            restore_paras['stage_name'],
                                            True)

        # evulate_imgs(
        #     nc_roi_paths, art_roi_paths, pv_roi_paths, nc_patches_paths, art_patches_paths, pv_patches_paths, labels,
        #     load_result=False, netname=restore_paras['netname'], model_path=restore_paras['model_path']
        # )
        evulate_imgs_batch_with_attributions(
            nc_roi_paths, art_roi_paths, pv_roi_paths, nc_patches_paths, art_patches_paths, pv_patches_paths, attrs,
            labels, model_path=restore_paras['model_path'], netname=restore_paras['netname']
        )
    else:
        nc_roi_paths, art_roi_paths, pv_roi_paths, nc_patches_paths, art_patches_paths, pv_patches_paths, labels = generate_paths(
            '/media/dl-box/HDD3/ld/Documents/datasets/IEEEonMedicalImage_Splited/JPG/0', restore_paras['stage_name'], True)

        # evulate_imgs(
        #     nc_roi_paths, art_roi_paths, pv_roi_paths, nc_patches_paths, art_patches_paths, pv_patches_paths, labels,
        #     load_result=False, netname=restore_paras['netname'], model_path=restore_paras['model_path']
        # )
        evulate_imgs_batch(
            nc_roi_paths, art_roi_paths, pv_roi_paths, nc_patches_paths, art_patches_paths, pv_patches_paths, labels,
            model_path=restore_paras['model_path'], netname=restore_paras['netname']
        )



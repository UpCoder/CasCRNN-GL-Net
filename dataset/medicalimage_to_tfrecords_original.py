# -*- coding=utf-8 -*-
# 直接使用原始密度值，非病灶区域置0
# Copyright 2015 Paul Balanca. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Converts Pascal VOC data to TFRecords file format with Example protos.

The raw Pascal VOC data set is expected to reside in JPEG files located in the
directory 'JPEGImages'. Similarly, bounding box annotations are supposed to be
stored in the 'Annotation directory'

This TensorFlow script converts the training and evaluation data into
a sharded data set consisting of 1024 and 128 TFRecord files, respectively.

Each validation TFRecord file contains ~500 records. Each training TFREcord
file contains ~1000 records. Each record within the TFRecord file is a
serialized Example proto. The Example proto contains the following fields:

    image/encoded: string containing JPEG encoded image in RGB colorspace
    image/height: integer, image height in pixels
    image/width: integer, image width in pixels
    image/channels: integer, specifying the number of channels, always 3
    image/format: string, specifying the format, always'JPEG'


    image/object/bbox/xmin: list of float specifying the 0+ human annotated
        bounding boxes
    image/object/bbox/xmax: list of float specifying the 0+ human annotated
        bounding boxes
    image/object/bbox/ymin: list of float specifying the 0+ human annotated
        bounding boxes
    image/object/bbox/ymax: list of float specifying the 0+ human annotated
        bounding boxes
    image/object/bbox/label: list of integer specifying the classification index.
    image/object/bbox/label_text: list of string descriptions.

Note that the length of xmin is identical to the length of xmax, ymin and ymax
for each example.
"""
import os
import sys
import random

import numpy as np
import tensorflow as tf

import xml.etree.ElementTree as ET
from glob import glob
from dataset.dataset_utils import int64_feature, float_feature, bytes_feature, np_array_feature
from medicalImage import read_mhd_image, compute_mean_img, get_filled_mask, get_liver_path, read_liver_mask
import cv2
import config
from utils.Tools import get_boundingbox
from convert2jpg import extract_patches, calculate_mask_attributes
from config import RANDOM_SEED


# Original dataset organisation.
# TFRecords convertion parameters.
SAMPLES_PER_FILES = 200

MEDICAL_LABELS = {
    'none': (0, 'Background'),
    'CYST': (1, 'Begin'),
    'FNH': (1, 'Begin'),
    'HCC': (1, 'Begin'),
    'HEM': (1, 'Begin'),
    'METS': (1, 'Begin'),
}
MEDICAL_LABELS_multi_category = {
    'none': (0, 'Background'),
    'CYST': (1, 'Begin'),
    'FNH': (2, 'Begin'),
    'HCC': (3, 'Begin'),
    'HEM': (4, 'Begin'),
    'METS': (5, 'Begin'),
}

def _convert_to_example(nc_roi_data, art_roi_data, pv_roi_data, nc_patch_data, art_patch_data, pv_patch_data, attr, label):
    """Build an Example proto for an image example.

    Args:
      image_data: string, JPEG encoding of RGB image;
      labels: list of integers, identifier for the ground truth;
      labels_text: list of strings, human-readable labels;
      bboxes: list of bounding boxes; each box is a list of integers;
          specifying [xmin, ymin, xmax, ymax]. All boxes are assumed to belong
          to the same label as the image label.
      shape: 3 integers, image shapes in pixels.
    Returns:
      Example proto
    """

    image_format = b'RAW'
    # print(np.shape(nc_roi_data), np.shape(art_roi_data), np.shape(pv_roi_data), np.shape(nc_patch_data),
    #       np.shape(art_patch_data), np.shape(pv_patch_data))
    nc_roi_h, nc_roi_w, _ = np.shape(nc_roi_data)
    art_roi_h, art_roi_w, _ = np.shape(art_roi_data)
    pv_roi_h, pv_roi_w, _ = np.shape(pv_roi_data)
    nc_patch_h, nc_patch_w, _ = np.shape(nc_patch_data)
    art_patch_h, art_patch_w, _ = np.shape(art_patch_data)
    pv_patch_h, pv_patch_w, _ = np.shape(pv_patch_data)
    nc_roi_data = np.asarray(nc_roi_data, np.float32)
    art_roi_data = np.asarray(art_roi_data, np.float32)
    pv_roi_data = np.asarray(pv_roi_data, np.float32)
    nc_patch_data = np.asarray(nc_patch_data, np.float32)
    art_patch_data = np.asarray(art_patch_data, np.float32)
    pv_patch_data = np.asarray(pv_patch_data, np.float32)
    # print('np_array_feature')
    # print('attrs is ', attrs)
    example = tf.train.Example(features=tf.train.Features(feature={
        'images/attrs': float_feature(list(attr)),
        'images/label': int64_feature(label),
        'images/nc_roi': np_array_feature(nc_roi_data),
        'images/nc_roi/shape': int64_feature([nc_roi_h, nc_roi_w, 3]),
        'images/art_roi': np_array_feature(art_roi_data),
        'images/art_roi/shape': int64_feature([art_roi_h, art_roi_w, 3]),
        'images/pv_roi': np_array_feature(pv_roi_data),
        'images/pv_roi/shape': int64_feature([pv_roi_h, pv_roi_w, 3]),
        'images/nc_patch': np_array_feature(nc_patch_data),
        'images/nc_patch/shape': int64_feature([nc_patch_h, nc_patch_w, 3]),
        'images/art_patch': np_array_feature(art_patch_data),
        'images/art_patch/shape': int64_feature([art_patch_h, art_patch_w, 3]),
        'images/pv_patch': np_array_feature(pv_patch_data),
        'images/pv_patch/shape': int64_feature([pv_patch_h, pv_patch_w, 3]),
        'images/format': bytes_feature(image_format)}))

    # 'image/height': int64_feature(shape[0]),
    # 'image/width': int64_feature(shape[1]),
    # 'image/channels': int64_feature(shape[2]),
    # 'image/shape': int64_feature(shape),
    # 'image/object/bbox/xmin': float_feature(xmin),
    # 'image/object/bbox/xmax': float_feature(xmax),
    # 'image/object/bbox/ymin': float_feature(ymin),
    # 'image/object/bbox/ymax': float_feature(ymax),
    # 'image/object/bbox/label': int64_feature(labels),
    # 'image/object/bbox/label_text': bytes_feature(labels_text),
    # 'image/object/bbox/difficult': int64_feature(difficult),
    # 'image/object/bbox/truncated': int64_feature(truncated),
    return example


def _add_to_tfrecord(nc_roi, art_roi, pv_roi, nc_patch, art_patch, pv_patch, attr, label, tfrecord_writer):
    """Loads data from image and annotations files and add them to a TFRecord.

    Args:
      dataset_dir: Dataset directory;
      name: Image name to add to the TFRecord;
      tfrecord_writer: The TFRecord writer to use for writing.
    """
    example = _convert_to_example(nc_roi, art_roi, pv_roi, nc_patch, art_patch, pv_patch, attr, label)
    tfrecord_writer.write(example.SerializeToString())


def _get_output_filename(output_dir, name, idx):
    return '%s/%s_%03d.tfrecord' % (output_dir, name, idx)


def run(dataset_dir, output_dir, name='voc_train', shuffling=True, stage_name='train', patch_size=5):
    """Runs the conversion operation.

    Args:
      dataset_dir: The dataset directory where the dataset is stored.
      output_dir: Output directory.
    """
    if not tf.gfile.Exists(dataset_dir):
        tf.gfile.MakeDirs(dataset_dir)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    # Dataset filenames, and shuffling.
    cur_dataset_dir = os.path.join(dataset_dir, stage_name)
    slice_names = os.listdir(cur_dataset_dir)
    nc_rois= []
    art_rois = []
    pv_rois= []
    nc_patches= []
    art_patches = []
    pv_patches= []
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
        # 避免因为patch size过大，而导致patch的个数为0
        if target_h < patch_size:
            target_h = patch_size + 1
        if target_w < patch_size:
            target_w = patch_size + 1
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

    # Process dataset files.
    i = 0
    fidx = 0
    while i < len(nc_rois):
        # Open new TFRecord file.
        tf_filename = _get_output_filename(output_dir, name, fidx)
        with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
            j = 0
            while i < len(nc_rois) and j < SAMPLES_PER_FILES:
                sys.stdout.write('\r>> Converting image %d/%d' % (i+1, len(nc_rois)))
                sys.stdout.flush()

                nc_roi = nc_rois[i]
                art_roi = art_rois[i]
                pv_roi = pv_rois[i]

                nc_patch = nc_patches[i]
                art_patch = art_patches[i]
                pv_patch = pv_patches[i]

                label = labels[i]
                attr = attrs[i]

                _add_to_tfrecord(nc_roi, art_roi, pv_roi, nc_patch, art_patch, pv_patch, attr, label, tfrecord_writer)

                i += 1
                j += 1
            fidx += 1

    # Finally, write the labels file:
    # labels_to_class_names = dict(zip(range(len(_CLASS_NAMES)), _CLASS_NAMES))
    # dataset_utils.write_label_file(labels_to_class_names, dataset_dir)
    print('\nFinished converting the Pascal VOC dataset!')


def process_single_file(dataset_dir, slice_name):
    cur_label = int(slice_name[-1])
    cur_slice_dir = os.path.join(dataset_dir, slice_name)

    nc_img_path = glob(os.path.join(cur_slice_dir, 'NC_Image*.mhd'))[0]
    art_img_path = glob(os.path.join(cur_slice_dir, 'ART_Image*.mhd'))[0]
    pv_img_path = glob(os.path.join(cur_slice_dir, 'PV_Image*.mhd'))[0]
    print(nc_img_path, art_img_path, pv_img_path)
    nc_img = np.squeeze(read_mhd_image(nc_img_path))
    art_img = np.squeeze(read_mhd_image(art_img_path))
    pv_img = np.squeeze(read_mhd_image(pv_img_path))

    nc_mask_path = glob(os.path.join(cur_slice_dir, 'NC_Mask*.mhd'))[0]
    art_mask_path = glob(os.path.join(cur_slice_dir, 'ART_Mask*.mhd'))[0]
    pv_mask_path = glob(os.path.join(cur_slice_dir, 'PV_Mask*.mhd'))[0]
    print(nc_mask_path, art_mask_path, pv_mask_path)
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
    print(nc_liver_mask_path, art_liver_mask_path, pv_liver_mask_path)
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
                                                                      patch_size=5)


if __name__ == '__main__':
    process_single_file('/home/dl-box/ld/Documents/datasets/IEEEonMedicalImage_Splited/0/train', '2298760_3920495_0_1_1')
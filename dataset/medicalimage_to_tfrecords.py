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
from dataset.dataset_utils import int64_feature, float_feature, bytes_feature


# Original dataset organisation.
# TFRecords convertion parameters.
RANDOM_SEED = 4242
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


def _process_image(nc_roi_path, art_roi_path, pv_roi_path, nc_patch_path, art_patch_path, pv_patch_path, attribute_flag=False):
    """Process a image and annotation file.

    Args:
      filename: string, path to an image file e.g., '/path/to/example.JPG'.
      coder: instance of ImageCoder to provide TensorFlow image coding utils.
    Returns:
      image_buffer: string, JPEG encoding of RGB image.
      height: integer, image height in pixels.
      width: integer, image width in pixels.
    """
    def _resolve_attribute_file(file_path):
        with open(file_path, 'r') as attr_file:
            lines = attr_file.readlines()
            line = lines[0]
            line_splited = line.split(' ')
            height = int(line_splited[0])
            width = int(line_splited[1])
            perimeter = int(line_splited[2])
            area = int(line_splited[3])
            circle_rate = float(line_splited[4])
            return height, width, perimeter, area, circle_rate
    if attribute_flag:
        from config import MEAN_ATTRS, MAX_ATTRS
        roi_dir_path = os.path.dirname(nc_roi_path)
        nc_attr_path = os.path.join(roi_dir_path, 'NC_attributes.txt')
        art_attr_path = os.path.join(roi_dir_path, 'ART_attributes.txt')
        pv_attr_path = os.path.join(roi_dir_path, 'PV_attributes.txt')
        nc_attrs = np.asarray(_resolve_attribute_file(nc_attr_path), np.float32)
        art_attrs = np.asarray(_resolve_attribute_file(art_attr_path), np.float32)
        pv_attrs = np.asarray(_resolve_attribute_file(pv_attr_path), np.float32)
        # nc_attrs = nc_attrs - MEAN_ATTRS
        # pv_attrs = pv_attrs - MEAN_ATTRS
        # art_attrs = art_attrs - MEAN_ATTRS
        nc_attrs /= MAX_ATTRS
        art_attrs /= MAX_ATTRS
        pv_attrs /= MAX_ATTRS
        attrs = np.concatenate([nc_attrs, art_attrs, pv_attrs], axis=-1)
        return tf.gfile.FastGFile(nc_roi_path).read(), tf.gfile.FastGFile(art_roi_path).read(), tf.gfile.FastGFile(
            pv_roi_path).read(), tf.gfile.FastGFile(nc_patch_path).read(), tf.gfile.FastGFile(
            art_patch_path).read(), tf.gfile.FastGFile(pv_patch_path).read(), attrs

    return tf.gfile.FastGFile(nc_roi_path).read(), tf.gfile.FastGFile(art_roi_path).read(), tf.gfile.FastGFile(
        pv_roi_path).read(), tf.gfile.FastGFile(nc_patch_path).read(), tf.gfile.FastGFile(
        art_patch_path).read(), tf.gfile.FastGFile(pv_patch_path).read()


def _convert_to_example(nc_roi_data, art_roi_data, pv_roi_data, nc_patch_data, art_patch_data, pv_patch_data, label, attribute_flag=False, attrs=None):
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
    if attribute_flag:
        image_format = b'JPEG'
        # print('attrs is ', attrs)
        example = tf.train.Example(features=tf.train.Features(feature={
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
            'images/attrs': float_feature(list(attrs)),
            'images/label': int64_feature(label),
            'images/nc_roi': bytes_feature(nc_roi_data),
            'images/art_roi': bytes_feature(art_roi_data),
            'images/pv_roi': bytes_feature(pv_roi_data),
            'images/nc_patch': bytes_feature(nc_patch_data),
            'images/art_patch': bytes_feature(art_patch_data),
            'images/pv_patch': bytes_feature(pv_patch_data),
            'images/format': bytes_feature(image_format)}))
    else:
        image_format = b'JPEG'
        example = tf.train.Example(features=tf.train.Features(feature={
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
                'images/label': int64_feature(label),
                'images/nc_roi': bytes_feature(nc_roi_data),
                'images/art_roi': bytes_feature(art_roi_data),
                'images/pv_roi': bytes_feature(pv_roi_data),
                'images/nc_patch': bytes_feature(nc_patch_data),
                'images/art_patch': bytes_feature(art_patch_data),
                'images/pv_patch': bytes_feature(pv_patch_data),
                'images/format': bytes_feature(image_format)}))
    return example


def _add_to_tfrecord(nc_roi_path, art_roi_path, pv_roi_path, nc_patch_path, art_patch_path, pv_patch_path, label, tfrecord_writer, attribute_flag=False):
    """Loads data from image and annotations files and add them to a TFRecord.

    Args:
      dataset_dir: Dataset directory;
      name: Image name to add to the TFRecord;
      tfrecord_writer: The TFRecord writer to use for writing.
    """
    if attribute_flag:
        nc_roi_data, art_roi_data, pv_roi_data, nc_patch_data, art_patch_data, pv_patch_data, attrs = \
            _process_image(nc_roi_path, art_roi_path, pv_roi_path, nc_patch_path, art_patch_path, pv_patch_path,
                           attribute_flag=True)
        example = _convert_to_example(nc_roi_data, art_roi_data, pv_roi_data, nc_patch_data, art_patch_data,
                                      pv_patch_data, label, attribute_flag, attrs)
    else:
        nc_roi_data, art_roi_data, pv_roi_data, nc_patch_data, art_patch_data, pv_patch_data = \
            _process_image(nc_roi_path, art_roi_path, pv_roi_path, nc_patch_path, art_patch_path, pv_patch_path)
        example = _convert_to_example(nc_roi_data, art_roi_data, pv_roi_data, nc_patch_data, art_patch_data, pv_patch_data,
                                      label)
    tfrecord_writer.write(example.SerializeToString())


def _get_output_filename(output_dir, name, idx):
    return '%s/%s_%03d.tfrecord' % (output_dir, name, idx)


def run(dataset_dir, output_dir, name='voc_train', shuffling=True, stage_name='train'):
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
    patch_size = 5
    nc_roi_paths = []
    art_roi_paths = []
    pv_roi_paths = []
    nc_patches_paths = []
    art_patches_paths = []
    pv_patches_paths = []
    labels = []
    for slice_name in slice_names:
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
              len(pv_patches_paths), len(labels))
    # Process dataset files.
    i = 0
    fidx = 0
    while i < len(nc_roi_paths):
        # Open new TFRecord file.
        tf_filename = _get_output_filename(output_dir, name, fidx)
        with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
            j = 0
            while i < len(nc_roi_paths) and j < SAMPLES_PER_FILES:
                sys.stdout.write('\r>> Converting image %d/%d' % (i+1, len(nc_roi_paths)))
                sys.stdout.flush()

                nc_roi_path = nc_roi_paths[i]
                art_roi_path = art_roi_paths[i]
                pv_roi_path = pv_roi_paths[i]

                nc_patch_path = nc_patches_paths[i]
                art_patch_path = art_patches_paths[i]
                pv_patch_path = pv_patches_paths[i]

                label = labels[i]

                _add_to_tfrecord(nc_roi_path, art_roi_path, pv_roi_path, nc_patch_path, art_patch_path, pv_patch_path,
                                 label, tfrecord_writer)

                i += 1
                j += 1
            fidx += 1

    # Finally, write the labels file:
    # labels_to_class_names = dict(zip(range(len(_CLASS_NAMES)), _CLASS_NAMES))
    # dataset_utils.write_label_file(labels_to_class_names, dataset_dir)
    print('\nFinished converting the Pascal VOC dataset!')


def run_with_attributes(dataset_dir, output_dir, name='voc_train', shuffling=True, stage_name='train'):
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
    patch_size = 5
    nc_roi_paths = []
    art_roi_paths = []
    pv_roi_paths = []
    nc_patches_paths = []
    art_patches_paths = []
    pv_patches_paths = []
    labels = []
    for slice_name in slice_names:
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
              len(pv_patches_paths), len(labels))
    # Process dataset files.
    i = 0
    fidx = 0
    while i < len(nc_roi_paths):
        # Open new TFRecord file.
        tf_filename = _get_output_filename(output_dir, name, fidx)
        with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
            j = 0
            while i < len(nc_roi_paths) and j < SAMPLES_PER_FILES:
                sys.stdout.write('\r>> Converting image %d/%d' % (i+1, len(nc_roi_paths)))
                sys.stdout.flush()

                nc_roi_path = nc_roi_paths[i]
                art_roi_path = art_roi_paths[i]
                pv_roi_path = pv_roi_paths[i]

                nc_patch_path = nc_patches_paths[i]
                art_patch_path = art_patches_paths[i]
                pv_patch_path = pv_patches_paths[i]

                label = labels[i]

                _add_to_tfrecord(nc_roi_path, art_roi_path, pv_roi_path, nc_patch_path, art_patch_path, pv_patch_path,
                                 label, tfrecord_writer, attribute_flag=True)

                i += 1
                j += 1
            fidx += 1

    # Finally, write the labels file:
    # labels_to_class_names = dict(zip(range(len(_CLASS_NAMES)), _CLASS_NAMES))
    # dataset_utils.write_label_file(labels_to_class_names, dataset_dir)
    print('\nFinished converting the Pascal VOC dataset!')

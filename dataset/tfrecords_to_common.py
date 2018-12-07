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
"""Provides data for the Pascal VOC Dataset (images + annotations).
"""
import os

import tensorflow as tf
from dataset import dataset_utils

slim = tf.contrib.slim

VOC_LABELS = {
    'none': (0, 'Background'),
    'aeroplane': (1, 'Vehicle'),
    'bicycle': (2, 'Vehicle'),
    'bird': (3, 'Animal'),
    'boat': (4, 'Vehicle'),
    'bottle': (5, 'Indoor'),
    'bus': (6, 'Vehicle'),
    'car': (7, 'Vehicle'),
    'cat': (8, 'Animal'),
    'chair': (9, 'Indoor'),
    'cow': (10, 'Animal'),
    'diningtable': (11, 'Indoor'),
    'dog': (12, 'Animal'),
    'horse': (13, 'Animal'),
    'motorbike': (14, 'Vehicle'),
    'person': (15, 'Person'),
    'pottedplant': (16, 'Indoor'),
    'sheep': (17, 'Animal'),
    'sofa': (18, 'Indoor'),
    'train': (19, 'Vehicle'),
    'tvmonitor': (20, 'Indoor'),
}


def get_split(split_name, dataset_dir, file_pattern, reader,
              split_to_sizes, items_to_descriptions, num_classes):
    """Gets a dataset tuple with instructions for reading Pascal VOC dataset.

    Args:
      split_name: A train/test split name.
      dataset_dir: The base directory of the dataset sources.
      file_pattern: The file pattern to use when matching the dataset sources.
        It is assumed that the pattern contains a '%s' string so that the split
        name can be inserted.
      reader: The TensorFlow reader type.

    Returns:
      A `Dataset` namedtuple.

    Raises:
        ValueError: if `split_name` is not a valid train/test split.
    """
    if split_name not in split_to_sizes:
        raise ValueError('split name %s was not recognized.' % split_name)
    print(split_name)
    # file_pattern = os.path.join(dataset_dir, file_pattern % split_name)
    file_pattern = os.path.join(dataset_dir, file_pattern)
    print(file_pattern)
    # Allowing None in the signature so that dataset_factory can use the default.
    if reader is None:
        reader = tf.TFRecordReader
    # Features in Pascal VOC TFRecords.
    # 'images/label': int64_feature(label),
    # 'images/nc_roi': bytes_feature(nc_roi_data),
    # 'images/art_roi': bytes_feature(art_roi_data),
    # 'images/pv_roi': bytes_feature(pv_roi_data),
    # 'images/nc_patch': bytes_feature(nc_patch_data),
    # 'images/art_patch': bytes_feature(art_patch_data),
    # 'images/pv_patch': bytes_feature(pv_patch_data),
    # 'images/format': bytes_feature(image_format)}))
    keys_to_features = {
        'images/nc_roi': tf.FixedLenFeature((), tf.string, default_value=''),
        'images/art_roi': tf.FixedLenFeature((), tf.string, default_value=''),
        'images/pv_roi': tf.FixedLenFeature((), tf.string, default_value=''),
        'images/nc_patch': tf.FixedLenFeature((), tf.string, default_value=''),
        'images/art_patch': tf.FixedLenFeature((), tf.string, default_value=''),
        'images/pv_patch': tf.FixedLenFeature((), tf.string, default_value=''),
        'images/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'images/label': tf.FixedLenFeature([1], tf.int64),
    }
    items_to_handlers = {
        'nc_roi': slim.tfexample_decoder.Image('images/nc_roi', 'images/format'),
        'art_roi': slim.tfexample_decoder.Image('images/art_roi', 'images/format'),
        'pv_roi': slim.tfexample_decoder.Image('images/pv_roi', 'images/format'),
        'nc_patch': slim.tfexample_decoder.Image('images/nc_patch', 'images/format'),
        'art_patch': slim.tfexample_decoder.Image('images/art_patch', 'images/format'),
        'pv_patch': slim.tfexample_decoder.Image('images/pv_patch', 'images/format'),
        'label': slim.tfexample_decoder.Tensor('images/label')
    }
    decoder = slim.tfexample_decoder.TFExampleDecoder(
        keys_to_features, items_to_handlers)

    labels_to_names = None
    if dataset_utils.has_labels(dataset_dir):
        labels_to_names = dataset_utils.read_label_file(dataset_dir)
    # else:
    #     labels_to_names = create_readable_names_for_imagenet_labels()
    #     dataset_utils.write_label_file(labels_to_names, dataset_dir)

    return slim.dataset.Dataset(
            data_sources=file_pattern,
            reader=reader,
            decoder=decoder,
            num_samples=split_to_sizes[split_name],
            items_to_descriptions=items_to_descriptions,
            num_classes=num_classes,
            labels_to_names=labels_to_names)


def get_split_with_attributes(split_name, dataset_dir, file_pattern, reader,
                              split_to_sizes, items_to_descriptions, num_classes):
    """Gets a dataset tuple with instructions for reading Pascal VOC dataset.

    Args:
      split_name: A train/test split name.
      dataset_dir: The base directory of the dataset sources.
      file_pattern: The file pattern to use when matching the dataset sources.
        It is assumed that the pattern contains a '%s' string so that the split
        name can be inserted.
      reader: The TensorFlow reader type.

    Returns:
      A `Dataset` namedtuple.

    Raises:
        ValueError: if `split_name` is not a valid train/test split.
    """
    if split_name not in split_to_sizes:
        raise ValueError('split name %s was not recognized.' % split_name)
    print(split_name)
    # file_pattern = os.path.join(dataset_dir, file_pattern % split_name)
    file_pattern = os.path.join(dataset_dir, file_pattern)
    print(file_pattern)
    # Allowing None in the signature so that dataset_factory can use the default.
    if reader is None:
        reader = tf.TFRecordReader
    # Features in Pascal VOC TFRecords.
    # 'images/label': int64_feature(label),
    # 'images/nc_roi': bytes_feature(nc_roi_data),
    # 'images/art_roi': bytes_feature(art_roi_data),
    # 'images/pv_roi': bytes_feature(pv_roi_data),
    # 'images/nc_patch': bytes_feature(nc_patch_data),
    # 'images/art_patch': bytes_feature(art_patch_data),
    # 'images/pv_patch': bytes_feature(pv_patch_data),
    # 'images/format': bytes_feature(image_format)}))
    keys_to_features = {
        'images/nc_roi': tf.FixedLenFeature((), tf.string, default_value=''),
        'images/art_roi': tf.FixedLenFeature((), tf.string, default_value=''),
        'images/pv_roi': tf.FixedLenFeature((), tf.string, default_value=''),
        'images/nc_patch': tf.FixedLenFeature((), tf.string, default_value=''),
        'images/art_patch': tf.FixedLenFeature((), tf.string, default_value=''),
        'images/pv_patch': tf.FixedLenFeature((), tf.string, default_value=''),
        'images/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'images/label': tf.FixedLenFeature([1], tf.int64),
        'images/attrs': tf.FixedLenFeature([15], tf.float32)
    }
    items_to_handlers = {
        'nc_roi': slim.tfexample_decoder.Image('images/nc_roi', 'images/format'),
        'art_roi': slim.tfexample_decoder.Image('images/art_roi', 'images/format'),
        'pv_roi': slim.tfexample_decoder.Image('images/pv_roi', 'images/format'),
        'nc_patch': slim.tfexample_decoder.Image('images/nc_patch', 'images/format'),
        'art_patch': slim.tfexample_decoder.Image('images/art_patch', 'images/format'),
        'pv_patch': slim.tfexample_decoder.Image('images/pv_patch', 'images/format'),
        'attrs': slim.tfexample_decoder.Tensor('images/attrs'),
        'label': slim.tfexample_decoder.Tensor('images/label')
    }
    decoder = slim.tfexample_decoder.TFExampleDecoder(
        keys_to_features, items_to_handlers)

    labels_to_names = None
    if dataset_utils.has_labels(dataset_dir):
        labels_to_names = dataset_utils.read_label_file(dataset_dir)
    # else:
    #     labels_to_names = create_readable_names_for_imagenet_labels()
    #     dataset_utils.write_label_file(labels_to_names, dataset_dir)

    return slim.dataset.Dataset(
            data_sources=file_pattern,
            reader=reader,
            decoder=decoder,
            num_samples=split_to_sizes[split_name],
            items_to_descriptions=items_to_descriptions,
            num_classes=num_classes,
            labels_to_names=labels_to_names)


def get_split_original(split_name, dataset_dir, file_pattern, reader,
                       split_to_sizes, items_to_descriptions, num_classes):
    """Gets a dataset tuple with instructions for reading Pascal VOC dataset.

    Args:
      split_name: A train/test split name.
      dataset_dir: The base directory of the dataset sources.
      file_pattern: The file pattern to use when matching the dataset sources.
        It is assumed that the pattern contains a '%s' string so that the split
        name can be inserted.
      reader: The TensorFlow reader type.

    Returns:
      A `Dataset` namedtuple.

    Raises:
        ValueError: if `split_name` is not a valid train/test split.
    """
    if split_name not in split_to_sizes:
        raise ValueError('split name %s was not recognized.' % split_name)
    print(split_name)
    # file_pattern = os.path.join(dataset_dir, file_pattern % split_name)
    file_pattern = os.path.join(dataset_dir, file_pattern)
    print(file_pattern)
    # Allowing None in the signature so that dataset_factory can use the default.
    if reader is None:
        reader = tf.TFRecordReader
    keys_to_features = {
        'images/nc_roi': tf.VarLenFeature(tf.float32),
        'images/nc_roi/shape': tf.FixedLenFeature([3], tf.int64),
        'images/art_roi': tf.VarLenFeature(tf.float32),
        'images/art_roi/shape': tf.FixedLenFeature([3], tf.int64),
        'images/pv_roi': tf.VarLenFeature(tf.float32),
        'images/pv_roi/shape': tf.FixedLenFeature([3], tf.int64),
        'images/nc_patch': tf.VarLenFeature(tf.float32),
        'images/nc_patch/shape': tf.FixedLenFeature([3], tf.int64),
        'images/art_patch': tf.VarLenFeature(tf.float32),
        'images/art_patch/shape': tf.FixedLenFeature([3], tf.int64),
        'images/pv_patch': tf.VarLenFeature(tf.float32),
        'images/pv_patch/shape': tf.FixedLenFeature([3], tf.int64),
        'images/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'images/label': tf.FixedLenFeature([1], tf.int64),
        'images/attrs': tf.FixedLenFeature([15], tf.float32)
    }
    # items_to_handlers = {
    #     'nc_roi': slim.tfexample_decoder.Image('images/nc_roi', 'images/format'),
    #     'art_roi': slim.tfexample_decoder.Image('images/art_roi', 'images/format'),
    #     'pv_roi': slim.tfexample_decoder.Image('images/pv_roi', 'images/format'),
    #     'nc_patch': slim.tfexample_decoder.Image('images/nc_patch', 'images/format'),
    #     'art_patch': slim.tfexample_decoder.Image('images/art_patch', 'images/format'),
    #     'pv_patch': slim.tfexample_decoder.Image('images/pv_patch', 'images/format'),
    #     'attrs': slim.tfexample_decoder.Tensor('images/attrs'),
    #     'label': slim.tfexample_decoder.Tensor('images/label')
    # }
    items_to_handlers = {
        'nc_roi': slim.tfexample_decoder.Tensor('images/nc_roi', 'images/nc_roi/shape'),
        'art_roi': slim.tfexample_decoder.Tensor('images/art_roi', 'images/art_roi/shape'),
        'pv_roi': slim.tfexample_decoder.Tensor('images/pv_roi', 'images/pv_roi/shape'),
        'nc_patch': slim.tfexample_decoder.Tensor('images/nc_patch', 'images/nc_patch/shape'),
        'art_patch': slim.tfexample_decoder.Tensor('images/art_patch', 'images/art_patch/shape'),
        'pv_patch': slim.tfexample_decoder.Tensor('images/pv_patch', 'images/pv_patch/shape'),
        'attrs': slim.tfexample_decoder.Tensor('images/attrs'),
        'label': slim.tfexample_decoder.Tensor('images/label')
    }
    decoder = slim.tfexample_decoder.TFExampleDecoder(
        keys_to_features, items_to_handlers)

    labels_to_names = None
    if dataset_utils.has_labels(dataset_dir):
        labels_to_names = dataset_utils.read_label_file(dataset_dir)
    # else:
    #     labels_to_names = create_readable_names_for_imagenet_labels()
    #     dataset_utils.write_label_file(labels_to_names, dataset_dir)

    return slim.dataset.Dataset(
            data_sources=file_pattern,
            reader=reader,
            decoder=decoder,
            num_samples=split_to_sizes[split_name],
            items_to_descriptions=items_to_descriptions,
            num_classes=num_classes,
            labels_to_names=labels_to_names)
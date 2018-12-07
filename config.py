from __future__ import print_function
from pprint import pprint
import numpy as np
from tensorflow.contrib.slim.python.slim.data import parallel_reader
import tensorflow as tf
import util

num_classes = 5
PATCH_IMAGE_WIDTH = 128
PATCH_IMAGE_HEIGHT = 128
ROI_IMAGE_WIDTH = 256
ROI_IMAGE_HEIGHT = 256
WEIGHT_DECAY = 0.0005
TYPENAMES2LABEL = {
    'CYST': 0,
    'FNH': 1,
    'HCC': 2,
    'HEM': 3,
    'METS': 4
}
TRAIN_GPU_ID = '3'
val_batch_size = 32
patch_size = 5
CHECKPOINT_DIR = '/home/dl-box/ld/PycharmProjects/GL_BD_LSTM/checkpoints'

MEAN_ATTRS = np.asarray([5.49446136e+01, 5.62638689e+01, 1.86754929e+02, 3.45037290e+03,
                         8.63892040e-01], np.float32)
MAX_ATTRS = np.asarray([236., 209., 712., 33974., 1.096961], np.float32)
NC_IMG_MEAN = np.asarray([30.016285, 58.861176, 30.016285], np.float32)
ART_IMG_MEAN = np.asarray([55.895214, 70.9138, 55.895214], np.float32)
PV_IMG_MEAN = np.asarray([ 65.929306, 110.245476,  65.929306], np.float32)
RANDOM_SEED = 4242
slim = tf.contrib.slim


def print_config(flags, dataset, save_dir=None, print_to_file=True):
    def do_print(stream=None):
        print(util.log.get_date_str(), file=stream)
        print('\n# =========================================================================== #', file=stream)
        print('# Training flags:', file=stream)
        print('# =========================================================================== #', file=stream)

        def print_ckpt(path):
            # ckpt = util.tf.get_latest_ckpt(path)
            # if ckpt is not None:
            #     print('Resume Training from : %s' % (ckpt), file=stream)
            #     return True
            return False

        if not print_ckpt(flags.train_dir):
            print_ckpt(flags.checkpoint_path)

        pprint(flags.__flags, stream=stream)

        print('\n# =========================================================================== #', file=stream)
        print('# pixel_link net parameters:', file=stream)
        print('# =========================================================================== #', file=stream)
        vars = globals()
        for key in vars:
            var = vars[key]
            if util.dtype.is_number(var) or util.dtype.is_str(var) or util.dtype.is_list(var) or util.dtype.is_tuple(
                    var):
                pprint('%s=%s' % (key, str(var)), stream=stream)

        print('\n# =========================================================================== #', file=stream)
        print('# Training | Evaluation dataset files:', file=stream)
        print('# =========================================================================== #', file=stream)
        data_files = parallel_reader.get_data_files(dataset.data_sources)
        pprint(sorted(data_files), stream=stream)
        print('', file=stream)

    do_print(None)

    if print_to_file:
        # Save to a text file as well.
        if save_dir is None:
            save_dir = flags.train_dir

        util.io.mkdir(save_dir)
        path = util.io.join_path(save_dir, 'training_config.txt')
        with open(path, "a") as out:
            do_print(out)

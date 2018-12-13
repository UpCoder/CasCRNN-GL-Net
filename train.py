# -*- coding=utf-8 -*-
# test code to make sure the ground truth calculation and data batch works well.

import numpy as np
import tensorflow as tf  # test
from tensorflow.python.ops import control_flow_ops

from networks import networks, networks_with_attrs
import util
import os
import config
os.environ['CUDA_VISIBLE_DEVICES']=config.TRAIN_GPU_ID
slim = tf.contrib.slim


# =========================================================================== #
# Checkpoint and running Flags
# =========================================================================== #
tf.app.flags.DEFINE_string('train_dir', None,
                           'the path to store checkpoints and eventfiles for summaries')

tf.app.flags.DEFINE_string('checkpoint_path', None,
                           'the path of pretrained model to be used. If there are checkpoints in train_dir, this config will be ignored.')

tf.app.flags.DEFINE_float('gpu_memory_fraction', -1,
                          'the gpu memory fraction to be used. If less than 0, allow_growth = True is used.')

tf.app.flags.DEFINE_integer('batch_size', None, 'The number of samples in each batch.')
tf.app.flags.DEFINE_integer('num_gpus', 1, 'The number of gpus can be used.')
tf.app.flags.DEFINE_integer('max_number_of_steps', 20000, 'The maximum number of training steps.')
tf.app.flags.DEFINE_integer('log_every_n_steps', 1, 'log frequency')
tf.app.flags.DEFINE_bool("ignore_missing_vars", True, '')
tf.app.flags.DEFINE_string('checkpoint_exclude_scopes', ['fuse_multi_phase', 'pixel_seg', 'multiscale*'],
                           'checkpoint_exclude_scopes')

# =========================================================================== #
# Optimizer configs.
# =========================================================================== #
tf.app.flags.DEFINE_float('learning_rate', 0.0001, 'learning rate.')
tf.app.flags.DEFINE_float('momentum', 0.9, 'The momentum for the MomentumOptimizer')
tf.app.flags.DEFINE_float('weight_decay', 0.0001, 'The weight decay on the model weights.')
tf.app.flags.DEFINE_bool('using_moving_average', True, 'Whether to use ExponentionalMovingAverage')
tf.app.flags.DEFINE_float('moving_average_decay', 0.9999, 'The decay rate of ExponentionalMovingAverage')

# =========================================================================== #
# I/O and preprocessing Flags.
# =========================================================================== #
tf.app.flags.DEFINE_integer(
    'num_readers', 1,
    'The number of parallel readers that read data from the dataset.')
tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 24,
    'The number of threads used to create the batches.')

# =========================================================================== #
# Dataset Flags.
# =========================================================================== #
tf.app.flags.DEFINE_string(
    'dataset_name', None, 'The name of the dataset to load.')
tf.app.flags.DEFINE_string(
    'dataset_split_name', 'train', 'The name of the train/test split.')
tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')
tf.app.flags.DEFINE_integer('train_image_width', 224, 'Train image size')
tf.app.flags.DEFINE_integer('train_image_height', 224, 'Train image size')
tf.app.flags.DEFINE_string('file_pattern', 'medicalimage*.tfrecord', 'the pattern of tfrecords files')
tf.app.flags.DEFINE_string('netname', 'res50', 'the name of network')
tf.app.flags.DEFINE_boolean(
    'attribute_flag', False, 'the flag represent whether use the attribution'
)
tf.app.flags.DEFINE_boolean(
    'clstm_flag', False, 'the flag represent whether use the clstm block'
)
tf.app.flags.DEFINE_boolean(
    'pretrained_flag', True, 'the flag represent whether use the pretrained model'
)
tf.app.flags.DEFINE_boolean(
    'centerloss_flag', True, 'the flag represent whether use the center loss model'
)
tf.app.flags.DEFINE_integer('VALIDATION_INTERVAL', 10, 'the interval of validation')
tf.app.flags.DEFINE_boolean(
    'global_branch_flag', True, 'the flag represent wheather use the global branch flag'
)
tf.app.flags.DEFINE_boolean(
    'local_branch_flag', True, 'the flag represent wheather use the local branch flag'
)

FLAGS = tf.app.flags.FLAGS

checkpoints_names = {
    'res50': 'resnet_v2_50.ckpt',
    'vgg16': 'vgg_16.ckpt'
}


def config_initialization():
    # image shape and feature layers shape inference
    image_shape = (FLAGS.train_image_height, FLAGS.train_image_width)

    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')

    tf.logging.set_verbosity(tf.logging.DEBUG)
    util.init_logger(
        log_file='log_train_pixel_link_%d_%d.log' % image_shape,
        log_path=FLAGS.train_dir, stdout=False, mode='a')

    batch_size = FLAGS.batch_size
    # batch_size_per_gpu = config.batch_size_per_gpu

    tf.summary.scalar('batch_size', batch_size)
    # tf.summary.scalar('batch_size_per_gpu', batch_size_per_gpu)

    util.proc.set_proc_name('ld_train_on' + '_' + FLAGS.dataset_name + '_GPU_' + config.TRAIN_GPU_ID)

    from dataset import tfrecords_to_medicalimage
    train_dataset = tfrecords_to_medicalimage.get_split(FLAGS.dataset_split_name, FLAGS.dataset_dir, FLAGS.file_pattern, None, FLAGS.attribute_flag)
    # dataset = dataset_factory.get_dataset(FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)
    config.print_config(FLAGS, train_dataset)
    val_dataset_dir = os.path.join(os.path.dirname(FLAGS.dataset_dir), 'val_tfrecords')
    if not os.path.exists(val_dataset_dir):
        val_dataset_dir = os.path.join(os.path.dirname(FLAGS.dataset_dir), 'val')
    if not os.path.exists(val_dataset_dir):
        print val_dataset_dir
        assert False
    val_dataset = tfrecords_to_medicalimage.get_split('val', val_dataset_dir, FLAGS.file_pattern, None,
                                                      FLAGS.attribute_flag)
    return train_dataset, val_dataset


def create_dataset_batch_queue_with_attributes(dataset, prefix='train'):
    from models.research.slim.preprocessing.vgg_preprocessing import preprocess_image_GL
    with tf.device('/cpu:0'):
        with tf.name_scope(prefix + '_' + FLAGS.dataset_name + '_data_provider'):
            provider = slim.dataset_data_provider.DatasetDataProvider(
                dataset,
                num_readers=FLAGS.num_readers,
                common_queue_capacity=1000 * FLAGS.batch_size,
                common_queue_min=700 * FLAGS.batch_size,
                shuffle=True)
            [nc_roi, art_roi, pv_roi, nc_patch, art_patch, pv_patch, label, attrs] = provider.get(
                ['nc_roi', 'art_roi', 'pv_roi', 'nc_patch', 'art_patch', 'pv_patch', 'label', 'attrs'])
        nc_roi = tf.cast(nc_roi, tf.float32)
        art_roi = tf.cast(art_roi, tf.float32)
        pv_roi = tf.cast(pv_roi, tf.float32)
        nc_patch = tf.cast(nc_patch, tf.float32)
        art_patch = tf.cast(art_patch, tf.float32)
        pv_patch = tf.cast(pv_patch, tf.float32)

        nc_roi = tf.identity(nc_roi, prefix + '_' + 'nc_roi')
        art_roi = tf.identity(art_roi, prefix + '_' + 'art_roi')
        pv_roi = tf.identity(pv_roi, prefix + '_' + 'pv_roi')
        nc_patch = tf.identity(nc_patch, prefix + '_' + 'nc_patch')
        art_patch = tf.identity(art_patch, prefix + '_' + 'art_patch')
        pv_patch = tf.identity(pv_patch, prefix + '_' + 'pv_patch')

        print('nc_roi is ', nc_roi)
        tf.summary.image(prefix + '_' + '/roi/nc/unpreprocess', tf.expand_dims(nc_roi, axis=0))
        tf.summary.image(prefix + '_' + '/roi/art/unpreprocess', tf.expand_dims(art_roi, axis=0))
        tf.summary.image(prefix + '_' + '/roi/pv/unpreprocess', tf.expand_dims(pv_roi, axis=0))
        # Pre-processing image, labels and bboxes.
        nc_roi, art_roi, pv_roi, nc_patch, art_patch, pv_patch = preprocess_image_GL(nc_roi, art_roi, pv_roi, nc_patch,
                                                                                     art_patch, pv_patch,
                                                                                     config.ROI_IMAGE_HEIGHT,
                                                                                     config.ROI_IMAGE_WIDTH,
                                                                                     config.PATCH_IMAGE_HEIGHT,
                                                                                     config.PATCH_IMAGE_WIDTH,
                                                                                     is_training=True)
        nc_roi = tf.identity(nc_roi, prefix + '_' + 'preprocessed_nc_roi')
        art_roi = tf.identity(art_roi, prefix + '_' + 'preprocessed_art_roi')
        pv_roi = tf.identity(pv_roi, prefix + '_' + 'preprocessed_pv_roi')
        nc_patch = tf.identity(nc_patch, prefix + '_' + 'preprocessed_nc_patch')
        art_patch = tf.identity(art_patch, prefix + '_' + 'preprocessed_art_patch')
        pv_patch = tf.identity(pv_patch, prefix + '_' + 'preprocessed_pv_patch')
        tf.summary.image(prefix + '_' + '/roi/nc/preprocess', tf.expand_dims(nc_roi, axis=0))
        tf.summary.image(prefix + '_' + '/roi/art/preprocess', tf.expand_dims(art_roi, axis=0))
        tf.summary.image(prefix + '_' + '/roi/pv/preprocess', tf.expand_dims(pv_roi, axis=0))
        print('nc roi art roi pv roi is ', nc_roi, art_roi, pv_roi, nc_patch, art_patch, pv_patch)
        nc_roi.set_shape([config.ROI_IMAGE_HEIGHT, config.ROI_IMAGE_WIDTH, 3])
        art_roi.set_shape([config.ROI_IMAGE_HEIGHT, config.ROI_IMAGE_WIDTH, 3])
        pv_roi.set_shape([config.ROI_IMAGE_HEIGHT, config.ROI_IMAGE_WIDTH, 3])
        nc_patch.set_shape([config.PATCH_IMAGE_HEIGHT, config.PATCH_IMAGE_WIDTH, 3])
        art_patch.set_shape([config.PATCH_IMAGE_HEIGHT, config.PATCH_IMAGE_WIDTH, 3])
        pv_patch.set_shape([config.PATCH_IMAGE_HEIGHT, config.PATCH_IMAGE_WIDTH, 3])
        # batch them
        with tf.name_scope(prefix + '_' + FLAGS.dataset_name + '_batch'):
            b_nc_roi, b_art_roi, b_pv_roi, b_nc_patch, b_art_patch, b_pv_patch, b_label, b_attrs = \
                tf.train.batch(
                    [nc_roi, art_roi, pv_roi, nc_patch, art_patch, pv_patch, label, attrs],
                    batch_size=FLAGS.batch_size,
                    num_threads=FLAGS.num_preprocessing_threads,
                    capacity=500)
        with tf.name_scope(prefix + '_' + FLAGS.dataset_name + '_prefetch_queue'):
            batch_queue = slim.prefetch_queue.prefetch_queue(
                [b_nc_roi, b_art_roi, b_pv_roi, b_nc_patch, b_art_patch, b_pv_patch, b_label, b_attrs],
                capacity=50)
    return batch_queue


def create_dataset_batch_queue(dataset):
    from models.research.slim.preprocessing.vgg_preprocessing import preprocess_image_GL
    with tf.device('/cpu:0'):
        with tf.name_scope(FLAGS.dataset_name + '_data_provider'):
            provider = slim.dataset_data_provider.DatasetDataProvider(
                dataset,
                num_readers=FLAGS.num_readers,
                common_queue_capacity=1000 * FLAGS.batch_size,
                common_queue_min=700 * FLAGS.batch_size,
                shuffle=True)
            [nc_roi, art_roi, pv_roi, nc_patch, art_patch, pv_patch, label] = provider.get(
                ['nc_roi', 'art_roi', 'pv_roi', 'nc_patch', 'art_patch', 'pv_patch', 'label'])
        nc_roi = tf.identity(nc_roi, 'nc_roi')
        art_roi = tf.identity(art_roi, 'art_roi')
        pv_roi = tf.identity(pv_roi, 'pv_roi')
        nc_patch = tf.identity(nc_patch, 'nc_patch')
        art_patch = tf.identity(art_patch, 'art_patch')
        pv_patch = tf.identity(pv_patch, 'pv_patch')

        print('nc_roi is ', nc_roi)
        tf.summary.image('roi/nc/unpreprocess', tf.expand_dims(nc_roi, axis=0))
        tf.summary.image('roi/art/unpreprocess', tf.expand_dims(art_roi, axis=0))
        tf.summary.image('roi/pv/unpreprocess', tf.expand_dims(pv_roi, axis=0))
        # Pre-processing image, labels and bboxes.
        nc_roi, art_roi, pv_roi, nc_patch, art_patch, pv_patch = preprocess_image_GL(nc_roi, art_roi, pv_roi, nc_patch,
                                                                                     art_patch, pv_patch,
                                                                                     config.ROI_IMAGE_HEIGHT,
                                                                                     config.ROI_IMAGE_WIDTH,
                                                                                     is_training=True)
        nc_roi = tf.identity(nc_roi, 'preprocessed_nc_roi')
        art_roi = tf.identity(art_roi, 'preprocessed_art_roi')
        pv_roi = tf.identity(pv_roi, 'preprocessed_pv_roi')
        nc_patch = tf.identity(nc_patch, 'preprocessed_nc_patch')
        art_patch = tf.identity(art_patch, 'preprocessed_art_patch')
        pv_patch = tf.identity(pv_patch, 'preprocessed_pv_patch')
        tf.summary.image('roi/nc/preprocess', tf.expand_dims(nc_roi, axis=0))
        tf.summary.image('roi/art/preprocess', tf.expand_dims(art_roi, axis=0))
        tf.summary.image('roi/pv/preprocess', tf.expand_dims(pv_roi, axis=0))

        # batch them
        with tf.name_scope(FLAGS.dataset_name + '_batch'):
            b_nc_roi, b_art_roi, b_pv_roi, b_nc_patch, b_art_patch, b_pv_patch, b_label = \
                tf.train.batch(
                    [nc_roi, art_roi, pv_roi, nc_patch, art_patch, pv_patch, label],
                    batch_size=FLAGS.batch_size,
                    num_threads=FLAGS.num_preprocessing_threads,
                    capacity=500)
        with tf.name_scope(FLAGS.dataset_name + '_prefetch_queue'):
            batch_queue = slim.prefetch_queue.prefetch_queue(
                [b_nc_roi, b_art_roi, b_pv_roi, b_nc_patch, b_art_patch, b_pv_patch, b_label],
                capacity=50)
    return batch_queue


def sum_gradients(clone_grads):
    averaged_grads = []
    for idx, grad_and_vars in enumerate(zip(*clone_grads)):
        grads = []
        var = grad_and_vars[0][1]
        try:
            for g, v in grad_and_vars:
                assert v == var
                grads.append(g)
            grad = tf.add_n(grads, name=v.op.name + '_summed_gradients')
        except:
            import pdb
            pdb.set_trace()
        averaged_grads.append((grad, v))
    return averaged_grads


def create_clones(batch_queue):
    with tf.device('/cpu:0'):
        global_step = slim.create_global_step()
        learning_rate = tf.constant(FLAGS.learning_rate, name='learning_rate')
        optimizer = tf.train.MomentumOptimizer(learning_rate,
                                               momentum=FLAGS.momentum, name='Momentum')

        tf.summary.scalar('learning_rate', learning_rate)
    # place clones
    pixel_link_loss = 0  # for summary only
    gradients = []
    # for clone_idx, gpu in enumerate(config.gpus):
        # do_summary = clone_idx == 0  # only summary on the first clone
        # reuse = clone_idx > 0
    with tf.variable_scope(tf.get_variable_scope()):
        b_nc_roi, b_art_roi, b_pv_roi, b_nc_patch, b_art_patch, b_pv_patch, b_label = batch_queue.dequeue()
        # build model and loss
        net = networks(b_nc_roi, b_art_roi, b_pv_roi, b_nc_patch, b_art_patch, b_pv_patch,
                       FLAGS.netname, True, num_classes=config.num_classes, batch_size=FLAGS.batch_size)
        # ce_loss, center_loss, global_loss, local_loss = net.build_loss(b_label)
        ce_loss, center_loss, global_loss, local_loss = net.build_loss(b_label)

        # gather losses
        losses = tf.get_collection(tf.GraphKeys.LOSSES)
        assert len(losses) == 4
        total_clone_loss = tf.add_n(losses)
        pixel_link_loss += total_clone_loss

        # gather regularization loss and add to clone_0 only
        # if clone_idx == 0:
        regularization_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        total_clone_loss = total_clone_loss + regularization_loss

        # compute clone gradients
        clone_gradients = optimizer.compute_gradients(total_clone_loss)
        gradients.append(clone_gradients)

    tf.summary.scalar('final cross entropy', ce_loss)
    tf.summary.scalar('center loss', center_loss)
    tf.summary.scalar('regularization_loss', regularization_loss)
    tf.summary.scalar('global cross entropy', global_loss)
    tf.summary.scalar('local cross entropy', local_loss)

    # add all gradients together
    # note that the gradients do not need to be averaged, because the average operation has been done on loss.
    averaged_gradients = sum_gradients(gradients)

    apply_grad_op = optimizer.apply_gradients(averaged_gradients, global_step=global_step)

    train_ops = [apply_grad_op]

    bn_update_op = util.tf.get_update_op()
    if bn_update_op is not None:
        train_ops.append(bn_update_op)

    # moving average
    if FLAGS.using_moving_average:
        tf.logging.info('using moving average in training, \
        with decay = %f' % (FLAGS.moving_average_decay))
        ema = tf.train.ExponentialMovingAverage(FLAGS.moving_average_decay)
        ema_op = ema.apply(tf.trainable_variables())
        with tf.control_dependencies([apply_grad_op]):  # ema after updating
            train_ops.append(tf.group(ema_op))

    train_op = control_flow_ops.with_dependencies(train_ops, pixel_link_loss, name='train_op')
    return train_op


def train_step_fn(sess, train_op, global_step, train_step_kwargs):
    """
    slim.learning.train_step():
      train_step_kwargs = {summary_writer:,should_log:, should_stop:}
    usage: slim.learning.train( train_op, logdir,
                                train_step_fn=train_step_fn,)
    """
    if hasattr(train_step_fn, 'step'):
        train_step_fn.step += 1  # or use global_step.eval(session=sess)
    else:
        train_step_fn.step = global_step.eval(sess)

    # calc training losses
    total_loss, should_stop = slim.learning.train_step(sess, train_op, global_step, train_step_kwargs)

    # validate on interval
    if train_step_fn.step % FLAGS.VALIDATION_INTERVAL == 0:
        if 'val_loss' in train_step_kwargs:
            val_ce_loss_value, val_gb_loss_value, val_lb_loss_value = sess.run(
                [train_step_kwargs['val_ce_loss'], train_step_kwargs['val_gb_loss'],
                 train_step_kwargs['val_lb_loss']])
            print(">> global step {}:    train={}   val_ce_loss={}  val_gb_loss={} val_lb_loss={}".format(
                train_step_fn.step,
                total_loss, val_ce_loss_value,
                val_gb_loss_value, val_lb_loss_value))
        else:
            print('val_loss not in keys')
    return [total_loss, should_stop]


def create_clones_with_attrs(train_batch_queue, val_batch_queue):
    with tf.device('/cpu:0'):

        global_step = slim.create_global_step()
        learning_rate = tf.constant(FLAGS.learning_rate, name='learning_rate')
        optimizer = tf.train.MomentumOptimizer(learning_rate,
                                               momentum=FLAGS.momentum, name='Momentum')

        tf.summary.scalar('learning_rate', learning_rate)
    # place clones
    pixel_link_loss = 0  # for summary only
    gradients = []
    # for clone_idx, gpu in enumerate(config.gpus):
        # do_summary = clone_idx == 0  # only summary on the first clone
        # reuse = clone_idx > 0
    with tf.variable_scope(tf.get_variable_scope()) as sc:
        b_nc_roi, b_art_roi, b_pv_roi, b_nc_patch, b_art_patch, b_pv_patch, b_label, b_attrs = train_batch_queue.dequeue()
        val_b_nc_roi, val_b_art_roi, val_b_pv_roi, val_b_nc_patch, val_b_art_patch, val_b_pv_patch, \
        val_b_label, val_b_attrs = val_batch_queue.dequeue()
        # build model and loss
        train_net = networks_with_attrs(b_nc_roi, b_art_roi, b_pv_roi, b_nc_patch, b_art_patch, b_pv_patch, b_attrs,
                                        FLAGS.netname, True, num_classes=config.num_classes, batch_size=FLAGS.batch_size,
                                        use_attribute_flag=FLAGS.attribute_flag, clstm_flag=FLAGS.clstm_flag,
                                        global_branch_flag=FLAGS.global_branch_flag,
                                        local_branch_flag=FLAGS.local_branch_flag)
        # ce_loss, center_loss, global_loss, local_loss = net.build_loss(b_label)
        centerloss_lambda = 1.0
        if not FLAGS.centerloss_flag:
            print('do not use the center loss')
            centerloss_lambda = 0.0
        ce_loss, center_loss, global_loss, local_loss, center_update_op = train_net.build_loss(b_label,
                                                                                               lambda_center_loss=centerloss_lambda)
        sc.reuse_variables()
        val_net = networks_with_attrs(val_b_nc_roi, val_b_art_roi, val_b_pv_roi, val_b_nc_patch, val_b_art_patch,
                                      val_b_pv_patch, val_b_attrs, FLAGS.netname, False, config.num_classes,
                                      batch_size=FLAGS.batch_size, use_attribute_flag=FLAGS.attribute_flag,
                                      clstm_flag=FLAGS.clstm_flag, global_branch_flag=FLAGS.global_branch_flag,
                                      local_branch_flag=FLAGS.local_branch_flag)
        val_ce_loss, val_center_loss, val_global_loss, val_local_loss = val_net.build_loss(val_b_label,
                                                                                           lambda_center_loss=centerloss_lambda,
                                                                                           add_to_collection=False)

        losses = tf.get_collection(tf.GraphKeys.LOSSES)
        # final logit + center loss + local + global
        losses_num = 2 + int(FLAGS.global_branch_flag) + int(FLAGS.local_branch_flag)
        assert len(losses) == losses_num
        total_clone_loss = tf.add_n(losses)
        pixel_link_loss += total_clone_loss

        regularization_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        total_clone_loss = total_clone_loss + regularization_loss

        clone_gradients = optimizer.compute_gradients(total_clone_loss)
        gradients.append(clone_gradients)

    tf.summary.scalar('final cross entropy', ce_loss)
    tf.summary.scalar('center loss', center_loss)
    tf.summary.scalar('regularization_loss', regularization_loss)
    tf.summary.scalar('global cross entropy', global_loss)
    tf.summary.scalar('local cross entropy', local_loss)
    tf.summary.scalar('val/final cross_entropy', val_ce_loss)
    tf.summary.scalar('val/center loss', val_center_loss)
    tf.summary.scalar('val/global cross entropy', val_global_loss)
    tf.summary.scalar('val/local cross entropy', val_local_loss)

    # add all gradients together
    # note that the gradients do not need to be averaged, because the average operation has been done on loss.
    averaged_gradients = sum_gradients(gradients)

    apply_grad_op = optimizer.apply_gradients(averaged_gradients, global_step=global_step)

    train_ops = [apply_grad_op, center_update_op]

    bn_update_op = util.tf.get_update_op()
    if bn_update_op is not None:
        train_ops.append(bn_update_op)

    # moving average
    if FLAGS.using_moving_average:
        tf.logging.info('using moving average in training, \
        with decay = %f' % (FLAGS.moving_average_decay))
        ema = tf.train.ExponentialMovingAverage(FLAGS.moving_average_decay)
        ema_op = ema.apply(tf.trainable_variables())
        with tf.control_dependencies([apply_grad_op]):  # ema after updating
            train_ops.append(tf.group(ema_op))

    train_op = control_flow_ops.with_dependencies(train_ops, pixel_link_loss, name='train_op')
    train_step_kwargs = {}
    train_step_kwargs['val_loss'] = val_ce_loss
    train_step_kwargs['val_gb_loss'] = val_global_loss
    train_step_kwargs['val_lb_loss'] = val_local_loss
    return train_op, None


def get_init_fn(checkpoint_path):

    vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    # print(vars_list)
    # assign_ops = []
    load_dict = {}
    for var in vars_list:
        # print('var_value:',var.value)
        vname = str(var.name)
        if vname.startswith('patch_based/') or vname.startswith('roi_based/'):
            if vname.startswith('patch_based/'):
                from_name = vname[12:]
            else:
                from_name = vname[10:]
        else:
            continue
        # print('from_name:',from_name)

        try:
            # from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
            # print_tensors_in_checkpoint_file(checkpoint_path)
            var_value = tf.train.load_variable(checkpoint_path, from_name)
            var_shape = var.get_shape().as_list()
            from_shape = np.shape(var_value)
            if np.sum(var_shape) != np.sum(from_shape):
                print('Shape not equal! ', vname, var_shape, '<---', from_name, from_shape)
                continue
            print(vname, '<---', from_name)
            load_dict[vname] = var_value
        except Exception, e:
            print('Skip, ', vname)
            continue
        # print('var_value:',var_value)
        # assign_ops.append(tf.assign(var, var_value))
    return slim.assign_from_values_fn(load_dict)


def train(train_op, train_step_kwargs=None):
    summary_op = tf.summary.merge_all()
    sess_config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    if FLAGS.gpu_memory_fraction < 0:
        sess_config.gpu_options.allow_growth = True
    elif FLAGS.gpu_memory_fraction > 0:
        sess_config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_memory_fraction
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, checkpoints_names[FLAGS.netname])
    if FLAGS.pretrained_flag:
        init_fn = get_init_fn(checkpoint_path)
    else:
        init_fn = None
    # init_fn = util.tf.get_init_fn(checkpoint_path=checkpoint_path, train_dir=FLAGS.train_dir,
    #                               ignore_missing_vars=FLAGS.ignore_missing_vars,
    #                               checkpoint_exclude_scopes=FLAGS.checkpoint_exclude_scopes)
    saver = tf.train.Saver(max_to_keep=500, write_version=2)
    save_interval_secs = 300
    if train_step_kwargs is None:
        slim.learning.train(
            train_op,
            logdir=FLAGS.train_dir,
            init_fn=init_fn,
            summary_op=summary_op,
            number_of_steps=FLAGS.max_number_of_steps,
            log_every_n_steps=FLAGS.log_every_n_steps,
            save_summaries_secs=30,
            saver=saver,
            save_interval_secs=save_interval_secs,
            session_config=sess_config
        )
    else:
        slim.learning.train(
            train_op,
            train_step_kwargs=train_step_kwargs,
            train_step_fn=train_step_fn,
            logdir=FLAGS.train_dir,
            init_fn=init_fn,
            summary_op=summary_op,
            number_of_steps=FLAGS.max_number_of_steps,
            log_every_n_steps=FLAGS.log_every_n_steps,
            save_summaries_secs=30,
            saver=saver,
            save_interval_secs=save_interval_secs,
            session_config=sess_config
        )


def main(_):
    train_dataset, val_dataset = config_initialization()
    # if FLAGS.attribute_flag:
    train_batch_queue = create_dataset_batch_queue_with_attributes(train_dataset)
    val_batch_queue = create_dataset_batch_queue_with_attributes(val_dataset, prefix='val')
    train_op, train_step_kwargs = create_clones_with_attrs(train_batch_queue, val_batch_queue)
    train(train_op, train_step_kwargs)
    # else:
    #     batch_queue = create_dataset_batch_queue(train_dataset)
    #     train_op = create_clones(batch_queue)
    #     train(train_op)


if __name__ == '__main__':
    tf.app.run()


# wait to do
# 三个slice，一个是原始的密度，一个是减去肝脏后的密度，一个是(PV-NC, ART-NC, PV-ART)
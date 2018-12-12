# -*- coding=utf-8 -*-
import tensorflow as tf
from glob import glob
import os
import numpy as np


def resolve_eventfile(eventfile_path, attribute_name):
    xs = []
    ys = []
    iterator = tf.train.summary_iterator(eventfile_path)
    while True:
        try:
            e = iterator.next()
            if e is None:
                break
            for v in e.summary.value:
                # print e.status
                # print v.tag
                if v.tag == attribute_name:
                    # print(e.step, v.simple_value)
                    xs.append(e.step)
                    ys.append(v.simple_value)

                # if v.tag == 'val/final_cross_entropy':
                #     val_xs.append(e.step)
                #     val_ys.append(v.simple_value)
        except Exception, e:
            print e.message
            break
    return np.asarray(xs, np.int32), np.asarray(ys, np.float32)


def resolve_eventfiles(eventfile_dir):
    paths = glob(os.path.join(eventfile_dir, 'events.out.tfevents*'))
    paths = sorted(paths)
    global_train_xs = []
    global_train_ys = []
    for path in paths:
        train_xs, train_ys = resolve_eventfile(path, 'final_cross_entropy')
        if len(global_train_xs) == 0:
            max_xs = 0
        else:
            max_xs = np.max(global_train_xs)
        # rint xs
        if np.sum(np.where(train_xs >= max_xs)) != 0:
            start_index = np.min(np.where(train_xs >= max_xs))
        else:
            start_index = len(xs)
        print start_index
        global_train_xs.extend(train_xs[start_index:])
        global_train_ys.extend(train_ys[start_index:])
    print global_train_xs
    print global_train_ys

    global_val_xs = []
    global_val_ys = []
    for path in paths:
        val_xs, val_ys = resolve_eventfile(path, 'val/final_cross_entropy')
        if len(global_val_xs) == 0:
            max_xs = 0
        else:
            max_xs = np.max(global_val_xs)
        # rint xs
        if np.sum(np.where(val_xs >= max_xs)) != 0:
            start_index = np.min(np.where(val_xs >= max_xs))
        else:
            start_index = len(xs)
        print start_index
        global_val_xs.extend(val_xs[start_index:])
        global_val_ys.extend(val_ys[start_index:])
    print global_val_xs
    print global_val_ys
    return global_train_xs, global_train_ys, global_val_xs, global_val_ys


def plot_lines(global_xs, global_ys, names, xaxis_name, yaxis_name, save_name):
    import matplotlib.pyplot as plt
    if len(global_ys) != len(global_xs):
        print('the length of xs is not equal to global ys')
        assert False
    if names is not None and len(global_xs) != len(names):
        print('the length of names is not equal to the global xs')
        assert False
    for xs, ys in zip(global_xs, global_ys):
        plt.plot(xs, ys, linewidth=1)
    plt.legend(names, loc='upper right')
    plt.xlabel(xaxis_name)
    plt.ylabel(yaxis_name)
    # plt.show()
    plt.savefig('./' + save_name)


if __name__ == '__main__':
    eventfile_dir = '/media/dl-box/HDD3/ld/PycharmProjects/GL_BD_LSTM/logs/1/res50_original_0.0001'
    xs, ys, xs_val, ys_val = resolve_eventfiles(eventfile_dir)
    eventfile_dir = '/media/dl-box/HDD3/ld/PycharmProjects/GL_BD_LSTM/logs/1/res50_original_without_pretrained'
    xs_wo_pretrained, ys_wo_pretrained, xs_wo_pretrained_val, ys_wo_pretrained_val = resolve_eventfiles(eventfile_dir)
    plot_lines(
        [xs, xs_val, xs_wo_pretrained, xs_wo_pretrained_val],
        [ys, ys_val, ys_wo_pretrained, ys_wo_pretrained_val],
        names=['res50 train', 'res50 val', 'res50 wo pretrained train', 'res50 wo pretrained val'],
        xaxis_name='step', yaxis_name='loss value', save_name='dataset1.svg')
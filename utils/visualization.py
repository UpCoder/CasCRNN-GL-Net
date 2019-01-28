# -*- coding=utf-8 -*-
import matplotlib.pyplot as plt
import os
from glob import glob
from dataset.medicalImage import read_mhd_image
import numpy as np
import collections


def visualize_pixel_distribution(input=None):
    '''
    统计各个phase的pixel数值分布
    :return: 
    '''
    if input is None:
        cur_dataset_dir = '/home/dl-box/ld/Documents/datasets/IEEEonMedicalImage_Splited/0/train'
        art_dict = {}
        nc_dict = {}
        pv_dict = {}
        all_nc_imgs = []
        all_art_imgs = []
        all_pv_imgs = []
        slice_names = os.listdir(cur_dataset_dir)
        for idx, slice_name in enumerate(slice_names):
            if slice_name.startswith('.DS'):
                continue
            print(idx, ' / ', len(slice_names))
            cur_label = int(slice_name[-1])
            cur_slice_dir = os.path.join(cur_dataset_dir, slice_name)

            nc_img_path = glob(os.path.join(cur_slice_dir, 'NC_Image*.mhd'))[0]
            art_img_path = glob(os.path.join(cur_slice_dir, 'ART_Image*.mhd'))[0]
            pv_img_path = glob(os.path.join(cur_slice_dir, 'PV_Image*.mhd'))[0]

            nc_img = read_mhd_image(nc_img_path)
            art_img = read_mhd_image(art_img_path)
            pv_img = read_mhd_image(pv_img_path)
            nc_img = nc_img.flatten()
            art_img = art_img.flatten()
            pv_img = pv_img.flatten()
            all_nc_imgs.extend(list(nc_img))
            all_art_imgs.extend(list(art_img))
            all_pv_imgs.extend(list(pv_img))
        nc_keys, nc_counts = np.unique(all_nc_imgs, return_counts=True)
        for idx, nc_key in enumerate(nc_keys):
            nc_dict[nc_key] = nc_counts[idx]

        art_keys, art_counts = np.unique(all_art_imgs, return_counts=True)
        for idx, art_key in enumerate(art_keys):
            art_dict[art_key] = art_counts[idx]

        pv_keys, pv_counts = np.unique(all_pv_imgs, return_counts=True)
        for idx, pv_key in enumerate(pv_keys):
            pv_dict[pv_key] = pv_counts[idx]
        np.save('./nc_distribution.npy', nc_dict)
        np.save('./art_distribution.npy', art_dict)
        np.save('./pv_distribution.npy', pv_dict)
    else:
        flag = 'whole'
        nc_dict = np.load('./nc_distribution.npy').item()
        art_dict = np.load('./art_distribution.npy').item()
        pv_dict = np.load('./pv_distribution.npy').item()
        nc_xs = np.asarray(nc_dict.keys(), np.int64)
        nc_ys = np.asarray(nc_dict.values(), np.int64)
        art_xs = np.asarray(art_dict.keys(), np.int64)
        art_ys = np.asarray(art_dict.values(), np.int64)
        pv_xs = np.asarray(pv_dict.keys(), np.int64)
        pv_ys = np.asarray(pv_dict.values(), np.int64)
        if flag == 'whole':
            r_value = -3000
            l_value = 2000
        else:
            r_value = -500
            l_value = 500
        index = np.argsort(nc_xs)
        nc_xs = nc_xs[index]
        nc_ys = nc_ys[index]
        nc_index = np.min(np.where(nc_xs > r_value))
        nc_xs = nc_xs[nc_index: ]
        nc_ys = nc_ys[nc_index: ]
        nc_index = np.min(np.where(nc_xs > l_value))
        nc_xs = nc_xs[:nc_index]
        nc_ys = nc_ys[:nc_index]
        print(nc_xs)
        print(nc_ys)
        index = np.argsort(art_xs)
        art_xs = art_xs[index]
        art_ys = art_ys[index]
        art_index = np.min(np.where(art_xs > r_value))
        art_xs = art_xs[art_index:]
        art_ys = art_ys[art_index:]
        art_index = np.min(np.where(art_xs > l_value))
        art_xs = art_xs[:art_index]
        art_ys = art_ys[:art_index]

        index = np.argsort(pv_xs)
        pv_xs = pv_xs[index]
        pv_ys = pv_ys[index]
        pv_index = np.min(np.where(pv_xs > r_value))
        pv_xs = pv_xs[pv_index: ]
        pv_ys = pv_ys[pv_index: ]

        pv_index = np.min(np.where(pv_xs > l_value))
        pv_xs = pv_xs[:pv_index]
        pv_ys = pv_ys[:pv_index]
        plt.bar(nc_xs, nc_ys, linewidth=1, color='r', alpha=0.5, linestyle=':')
        plt.bar(art_xs, art_ys, linewidth=1, color='g', alpha=0.5, linestyle='--')
        plt.bar(pv_xs, pv_ys, linewidth=1, color='b', alpha=0.5)
        plt.legend(['NC', 'ART', 'PV'], loc='upper right')
        # plt.xlim((-3000, 3000))
        # plt.xticks([])
        plt.yticks([])
        plt.savefig('./' + 'pixel_distribution_' + flag + '.svg')


def visualize_patch_size():
    xs = [3, 5, 7, 9]
    ys_dataset1 = [87.33, 83.57, 84.19, 85.11]
    ys_dataset2 = [79.19, 83.44, 84.09, 76.96]
    ys_mean = [83.26, 83.50, 84.14, 81.03]
    names = ['dataset1', 'dataset2', 'mean']
    # plt.plot(xs, ys_dataset1, linewidth=1)
    # plt.plot(xs, ys_dataset2, linewidth=1)
    plt.plot(xs, ys_mean, linewidth=1)
    plt.plot(xs, ys_mean, 'b+')
    # plt.legend(names, loc='upper right')
    plt.xlabel('patch_size', fontsize=15)
    plt.ylabel('accuracy(%)', fontsize=15)
    plt.ylim((70, 100))
    # plt.show()
    plt.savefig('./' + 'patch_size.svg')


def visualize_alpha():
    alpha = [0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
    mean_value = [78.44, 82.10, 86.52, 87.75, 83.50, 82.88, 83.37]

    plt.plot(alpha, mean_value)
    plt.plot(alpha, mean_value, 'b+')
    plt.xlabel('alpha', fontsize=15)
    plt.ylabel('accuracy(%)', fontsize=15)
    plt.ylim((70, 100))
    plt.savefig('./' + 'alpha.svg')


def visualize_lambda():
    lambda_list = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
    mean_value = [77.06, 82.53, 85.53, 85.00, 87.75, 85.48, 85.94]
    plt.plot(lambda_list, mean_value)
    plt.plot(lambda_list, mean_value, 'b+')
    plt.xlabel('lambda', fontsize=15)
    plt.ylabel('accuracy(%)', fontsize=15)
    plt.ylim((70, 100))
    plt.savefig('./' + 'lambda.svg')


if __name__ == '__main__':
    # visualize_patch_size()
    # visualize_alpha()
    # visualize_lambda()

    visualize_pixel_distribution(True)
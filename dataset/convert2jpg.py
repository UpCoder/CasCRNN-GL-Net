# -*- coding=utf-8 -*-
import os
from glob import glob
from medicalImage import read_mhd_image
import numpy as np
from xml.dom.minidom import Document
import cv2
from utils.Tools import get_boundingbox
import math

def MICCAI2018_Iterator(image_dir, execute_func, *parameters):
    '''
    遍历MICCAI2018文件夹的框架
    :param execute_func:
    :return:
    '''
    for sub_name in ['train', 'val', 'test']:
        names = os.listdir(os.path.join(image_dir, sub_name))
        for name in names:
            cur_slice_dir = os.path.join(image_dir, sub_name, name)
            execute_func(cur_slice_dir, sub_name, *parameters)


def extract_patches(nc_roi, art_roi, pv_roi, patch_size):
    width, height, _ = np.shape(nc_roi)
    nc_patches = []
    art_patches = []
    pv_patches = []
    patch_step = 1
    if width * height >= 400:
        patch_step = int(math.sqrt(width * height / 100))
    for i in range(patch_size / 2, width - patch_size / 2, patch_step):
        for j in range(patch_size / 2, height - patch_size / 2, patch_step):
            cur_nc_patch = nc_roi[i - patch_size / 2:i + patch_size / 2 + 1,
                        j - patch_size / 2: j + patch_size / 2 + 1, :]
            cur_art_patch = art_roi[i - patch_size / 2:i + patch_size / 2 + 1,
                           j - patch_size / 2: j + patch_size / 2 + 1, :]
            cur_pv_patch = pv_roi[i - patch_size / 2:i + patch_size / 2 + 1,
                           j - patch_size / 2: j + patch_size / 2 + 1, :]
            nc_patches.append(cur_nc_patch)
            art_patches.append(cur_art_patch)
            pv_patches.append(cur_pv_patch)
    return nc_patches, art_patches, pv_patches


def get_window_center_info(load_path = '/media/dl-box/HDCZ-UT/datasets/MedicalImage/window_center_info.txt'):
    with open(load_path, 'r') as wc_file:
        lines = wc_file.readlines()
    window_center_dict = {}
    for line in lines:
        line_splited = line.split(' ')
        # print line
        # if line_splited[0] in window_center_dict.keys():
        #     print('there are dulipcate')
        #     assert False
        window_center_dict[line_splited[0]] = [int(ele) for ele in line_splited[1:]]
    return window_center_dict


def delete_useless(slice_dir, stage_name, save_dir):
    patient_name = os.path.basename(slice_dir).split('_')[0]
    cur_save_dir = os.path.join(save_dir, stage_name, os.path.basename(slice_dir))
    window_center_dict = get_window_center_info()

    if not os.path.exists(cur_save_dir):
        os.makedirs(cur_save_dir)
    nc_image_paths = glob(os.path.join(slice_dir, '%s_Image*.mhd' % 'NC'))
    art_image_paths = glob(os.path.join(slice_dir, '%s_Image*.mhd' % 'ART'))
    pv_image_paths = glob(os.path.join(slice_dir, '%s_Image*.mhd' % 'PV'))

    nc_mask_paths = glob(os.path.join(slice_dir, '%s_Mask*.mhd' % 'NC'))
    art_mask_paths = glob(os.path.join(slice_dir, '%s_Mask*.mhd' % 'ART'))
    pv_mask_paths = glob(os.path.join(slice_dir, '%s_Mask*.mhd' % 'PV'))

    nc_image_path = nc_image_paths[0]
    art_image_path = art_image_paths[0]
    pv_image_path = pv_image_paths[0]

    nc_mask_path = nc_mask_paths[0]
    art_mask_path = art_mask_paths[0]
    pv_mask_path = pv_mask_paths[0]
    save_paths = [nc_image_path, art_image_path, pv_image_path, nc_mask_path, art_mask_path, pv_mask_path]

    all_mhd_paths = glob(os.path.join(slice_dir, '*.mhd'))
    for mhd_path in all_mhd_paths:
        if mhd_path in save_paths:
            continue
        os.remove(mhd_path)
        raw_path = mhd_path.replace('.mhd', '.raw')
        os.remove(raw_path)
    mask_expand_paths = glob(os.path.join(slice_dir, '*Expand.raw'))
    for mask_expand_path in mask_expand_paths:
        os.remove(mask_expand_path)
    npy_paths = glob(os.path.join(slice_dir, '*.npy'))
    for npy_path in npy_paths:
        os.remove(npy_path)
    bin_paths = glob(os.path.join(slice_dir, '*.bin'))
    for bin_path in bin_paths:
        os.remove(bin_path)
    png_paths = glob(os.path.join(slice_dir, '*.png'))
    for png_path in png_paths:
        os.remove(png_path)


def calculate_mask_attributes(mask):
    min_xs, max_xs, min_ys, max_ys = get_boundingbox(mask)
    from medicalImage import fill_region
    edge = cv2.Canny(mask, 1, 1)
    perimeter = np.sum(edge >= 200)
    area = np.sum(fill_region(mask)) * 1.0
    circle_metric = 4 * np.pi * area / (perimeter * perimeter * 1.0)
    return (max_xs - min_xs), (max_ys - min_ys), perimeter, area, circle_metric, edge

def convert2jpg_multiphase(slice_dir, stage_name, save_dir):
    patient_name = os.path.basename(slice_dir).split('_')[0]
    cur_save_dir = os.path.join(save_dir, stage_name, os.path.basename(slice_dir))
    window_center_dict = get_window_center_info()

    if not os.path.exists(cur_save_dir):
        os.makedirs(cur_save_dir)
    nc_image_paths = glob(os.path.join(slice_dir, '%s_Image*.mhd' % 'NC'))
    art_image_paths = glob(os.path.join(slice_dir, '%s_Image*.mhd' % 'ART'))
    pv_image_paths = glob(os.path.join(slice_dir, '%s_Image*.mhd' % 'PV'))

    nc_mask_paths = glob(os.path.join(slice_dir, '%s_Mask*.mhd' % 'NC'))
    art_mask_paths = glob(os.path.join(slice_dir, '%s_Mask*.mhd' % 'ART'))
    pv_mask_paths = glob(os.path.join(slice_dir, '%s_Mask*.mhd' % 'PV'))
    # for nc_mask_path in nc_mask_paths:
    #     os.remove(nc_mask_path)
    # for art_mask_path in art_mask_paths:
    #     os.remove(art_mask_path)
    # for pv_mask_path in pv_mask_paths:
    #     os.remove(pv_mask_path)

    # if len(nc_image_paths) == 1 and len(art_image_paths) == 1 and len(pv_image_paths) == 1:
    #     print('OK', slice_dir)
    # else:
    #     print('Error', slice_dir, len(nc_image_paths), len(art_image_paths), len(pv_image_paths), nc_image_paths)
    #
    # if len(nc_mask_paths) == 1 and len(art_mask_paths) == 1 and len(pv_mask_paths) == 1:
    #     print('OK', slice_dir)
    # else:
    #     print('Error', slice_dir, len(nc_mask_paths), len(art_mask_paths), len(pv_mask_paths), nc_mask_paths)

    nc_image_path = nc_image_paths[0]
    art_image_path = art_image_paths[0]
    pv_image_path = pv_image_paths[0]

    nc_mask_path = nc_mask_paths[0]
    art_mask_path = art_mask_paths[0]
    pv_mask_path = pv_mask_paths[0]

    nc_image = np.squeeze(read_mhd_image(nc_image_path))
    art_image = np.squeeze(read_mhd_image(art_image_path))
    pv_image = np.squeeze(read_mhd_image(pv_image_path))
    nc_mask = np.asarray(np.squeeze(read_mhd_image(nc_mask_path)) == 1, np.uint8)
    art_mask = np.asarray(np.squeeze(read_mhd_image(art_mask_path)) == 1, np.uint8)
    pv_mask = np.asarray(np.squeeze(read_mhd_image(pv_mask_path)) == 1, np.uint8)

    def preprocessing_mhd(image, windows=[-350, 300]):
        image = np.asarray(image, np.float32)
        image[image < windows[0]] = windows[0]
        image[image > windows[1]] = windows[1]
        # image = image - np.mean(image)
        min_v = np.min(image)
        max_v = np.max(image)
        interv = max_v - min_v
        image = (image - min_v) / interv
        return image * 255.0

    def bounding_box(mask):
        xs, ys = np.where(mask == 1)
        return np.min(xs), np.min(ys), np.max(xs), np.max(ys)

    def calculate_mask_attributes(mask):
        min_xs, min_ys, max_xs, max_ys = bounding_box(mask)
        from medicalImage import fill_region
        edge = cv2.Canny(mask, 1, 1)
        perimeter = np.sum(edge >= 200)
        area = np.sum(fill_region(mask)) * 1.0
        circle_metric = 4 * np.pi * area / (perimeter * perimeter * 1.0)
        return (max_xs - min_xs), (max_ys - min_ys), perimeter, area, circle_metric, edge

    height_nc, width_nc, perimeter_nc, area_nc, circle_metric_nc, edge_nc = calculate_mask_attributes(nc_mask)
    with open(os.path.join(cur_save_dir, 'NC_attributes.txt'), 'w') as nc_attr_file:
        line = '%d %d %d %d %.6f\n' % (height_nc, width_nc, perimeter_nc, area_nc, circle_metric_nc)
        nc_attr_file.writelines([line])
        nc_attr_file.close()

    height_art, width_art, perimeter_art, area_art, circle_metric_art, edge_art = calculate_mask_attributes(art_mask)
    with open(os.path.join(cur_save_dir, 'ART_attributes.txt'), 'w') as art_attr_file:
        line = '%d %d %d %d %.6f\n' % (height_art, width_art, perimeter_art, area_art, circle_metric_art)
        art_attr_file.writelines([line])
        art_attr_file.close()
    height_pv, width_pv, perimeter_pv, area_pv, circle_metric_pv, edge_pv = calculate_mask_attributes(pv_mask)
    with open(os.path.join(cur_save_dir, 'PV_attributes.txt'), 'w') as pv_attr_file:
        line = '%d %d %d %d %.6f\n' % (height_pv, width_pv, perimeter_pv, area_pv, circle_metric_pv)
        pv_attr_file.writelines([line])
        pv_attr_file.close()

    # print('edge_nc: ', np.min(edge_nc), np.max(edge_nc), np.unique(edge_nc))
    # print(height_nc, width_nc, perimeter_nc, area_nc, circle_metric_nc, cur_save_dir[-1])
    cv2.imwrite(os.path.join(cur_save_dir, 'NC_EDGE.jpg'), edge_nc)

    if patient_name in window_center_dict.keys():
        cur_window_center = window_center_dict[patient_name]
    else:
        cur_window_center = [-70, 180, -70, 180, -70, 180]

    # print('cur_window_center is ', cur_window_center)
    nc_image = preprocessing_mhd(nc_image, [cur_window_center[0], cur_window_center[1]])
    art_image = preprocessing_mhd(art_image, [cur_window_center[2], cur_window_center[3]])
    pv_image = preprocessing_mhd(pv_image, [cur_window_center[4], cur_window_center[5]])

    nc_min_xs, nc_min_ys, nc_max_xs, nc_max_ys = bounding_box(nc_mask)
    art_min_xs, art_min_ys, art_max_xs, art_max_ys = bounding_box(art_mask)
    pv_min_xs, pv_min_ys, pv_max_xs, pv_max_ys = bounding_box(pv_mask)

    nc_roi = nc_image[nc_min_xs: nc_max_xs, nc_min_ys: nc_max_ys]
    art_roi = art_image[art_min_xs: art_max_xs, art_min_ys: art_max_ys]
    pv_roi = pv_image[pv_min_xs: pv_max_xs, pv_min_ys: pv_max_ys]
    nc_roi = np.concatenate(
        [np.expand_dims(nc_roi, axis=2), np.expand_dims(nc_roi, axis=2), np.expand_dims(nc_roi, axis=2)], axis=2)
    art_roi = np.concatenate(
        [np.expand_dims(art_roi, axis=2), np.expand_dims(art_roi, axis=2), np.expand_dims(art_roi, axis=2)], axis=2)
    pv_roi = np.concatenate(
        [np.expand_dims(pv_roi, axis=2), np.expand_dims(pv_roi, axis=2), np.expand_dims(pv_roi, axis=2)], axis=2)
    # 将所有的病灶resize到最大的那个病灶上面
    nc_h, nc_w, _ = np.shape(nc_roi)
    art_h, art_w, _ = np.shape(art_roi)
    pv_h, pv_w, _ = np.shape(pv_roi)
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
    nc_roi_resized = cv2.resize(nc_roi, (target_h, target_w))
    art_roi_resized = cv2.resize(art_roi, (target_h, target_w))
    pv_roi_resized = cv2.resize(pv_roi, (target_h, target_w))
    patch_size = 5
    nc_patches, art_patches, pv_patches = extract_patches(nc_roi_resized, art_roi_resized, pv_roi_resized, patch_size)

    nc_roi_save_path = os.path.join(cur_save_dir, 'NC.jpg')
    art_roi_save_path = os.path.join(cur_save_dir, 'ART.jpg')
    pv_roi_save_path = os.path.join(cur_save_dir, 'PV.jpg')
    cv2.imwrite(nc_roi_save_path, nc_roi_resized)
    cv2.imwrite(art_roi_save_path, art_roi_resized)
    cv2.imwrite(pv_roi_save_path, pv_roi_resized)
    patch_dir = os.path.join(cur_save_dir, '%dx%d' % (patch_size, patch_size))
    if not os.path.exists(patch_dir):
        os.mkdir(patch_dir)
    for idx, (nc_patch, art_patch, pv_patch) in enumerate(zip(nc_patches, art_patches, pv_patches)):
        nc_patch_save_path = os.path.join(patch_dir, '%d_NC.jpg' % idx)
        art_patch_save_path = os.path.join(patch_dir, '%d_ART.jpg' % idx)
        pv_patch_save_path = os.path.join(patch_dir, '%d_PV.jpg' % idx)
        cv2.imwrite(nc_patch_save_path, nc_patch)
        cv2.imwrite(art_patch_save_path, art_patch)
        cv2.imwrite(pv_patch_save_path, pv_patch)
    # print(slice_dir, 'num of patch is ', len(nc_patches))


def convert2jpg_multiphase_with_liver(slice_dir, stage_name, save_dir):
    if os.path.basename(slice_dir).startswith('.DS'):
        return

    def preprocessing_mhd(image, windows=[-350, 300]):
        image = np.asarray(image, np.float32)
        image[image < windows[0]] = windows[0]
        image[image > windows[1]] = windows[1]
        # image = image - np.mean(image)
        min_v = np.min(image)
        max_v = np.max(image)
        interv = max_v - min_v
        image = (image - min_v) / interv
        return image * 255.0

    def bounding_box(mask):
        xs, ys = np.where(mask == 1)
        return np.min(xs), np.min(ys), np.max(xs), np.max(ys)


    def _get_liver_path(cur_slice_dir, phase_name):
        paths = glob(os.path.join(cur_slice_dir, str(phase_name).lower() + '_liver_mask.mhd'))
        if len(paths) == 1:
            return paths[0]
        paths = glob(os.path.join(cur_slice_dir, str(phase_name).upper() + '_Liver_Mask.mhd'))
        if len(paths) == 1:
            return paths[0]
        print('Error ', cur_slice_dir, phase_name)
        assert False

    def _read_liver_mask(mask_path, size=16):
        liver_mask = np.squeeze(read_mhd_image(mask_path))
        min_xs, min_ys, max_xs, max_ys = bounding_box(liver_mask)
        center_x = min_xs + (max_xs - min_xs) // 2
        center_y = min_ys + (max_ys - min_ys) // 2
        new_min_xs = center_x - size // 2
        new_max_xs = center_x + size // 2
        new_min_ys = center_y - size // 2
        new_max_ys = center_y + size // 2
        return new_min_xs, new_min_ys, new_max_xs, new_max_ys

    patient_name = os.path.basename(slice_dir).split('_')[0]
    cur_save_dir = os.path.join(save_dir, stage_name, os.path.basename(slice_dir))
    print cur_save_dir
    window_center_dict = get_window_center_info()

    if not os.path.exists(cur_save_dir):
        os.makedirs(cur_save_dir)
    nc_image_paths = glob(os.path.join(slice_dir, '%s_Image*.mhd' % 'NC'))
    art_image_paths = glob(os.path.join(slice_dir, '%s_Image*.mhd' % 'ART'))
    pv_image_paths = glob(os.path.join(slice_dir, '%s_Image*.mhd' % 'PV'))

    nc_mask_paths = glob(os.path.join(slice_dir, '%s_Mask*.mhd' % 'NC'))
    art_mask_paths = glob(os.path.join(slice_dir, '%s_Mask*.mhd' % 'ART'))
    pv_mask_paths = glob(os.path.join(slice_dir, '%s_Mask*.mhd' % 'PV'))
    # for nc_mask_path in nc_mask_paths:
    #     os.remove(nc_mask_path)
    # for art_mask_path in art_mask_paths:
    #     os.remove(art_mask_path)
    # for pv_mask_path in pv_mask_paths:
    #     os.remove(pv_mask_path)

    # if len(nc_image_paths) == 1 and len(art_image_paths) == 1 and len(pv_image_paths) == 1:
    #     print('OK', slice_dir)
    # else:
    #     print('Error', slice_dir, len(nc_image_paths), len(art_image_paths), len(pv_image_paths), nc_image_paths)
    #
    # if len(nc_mask_paths) == 1 and len(art_mask_paths) == 1 and len(pv_mask_paths) == 1:
    #     print('OK', slice_dir)
    # else:
    #     print('Error', slice_dir, len(nc_mask_paths), len(art_mask_paths), len(pv_mask_paths), nc_mask_paths)
    if len(nc_image_paths) == 0:
        print slice_dir
        assert False
    nc_image_path = nc_image_paths[0]
    art_image_path = art_image_paths[0]
    pv_image_path = pv_image_paths[0]

    nc_mask_path = nc_mask_paths[0]
    art_mask_path = art_mask_paths[0]
    pv_mask_path = pv_mask_paths[0]

    nc_liver_mask_path = _get_liver_path(slice_dir, 'nc')
    art_liver_mask_path = _get_liver_path(slice_dir, 'art')
    pv_liver_mask_path = _get_liver_path(slice_dir, 'pv')
    nc_liver_min_xs, nc_liver_min_ys, nc_liver_max_xs, nc_liver_max_ys = _read_liver_mask(nc_liver_mask_path)
    art_liver_min_xs, art_liver_min_ys, art_liver_max_xs, art_liver_max_ys = _read_liver_mask(art_liver_mask_path)
    pv_liver_min_xs, pv_liver_min_ys, pv_liver_max_xs, pv_liver_max_ys = _read_liver_mask(pv_liver_mask_path)


    nc_image = np.squeeze(read_mhd_image(nc_image_path))
    art_image = np.squeeze(read_mhd_image(art_image_path))
    pv_image = np.squeeze(read_mhd_image(pv_image_path))
    nc_mask = np.asarray(np.squeeze(read_mhd_image(nc_mask_path)) == 1, np.uint8)
    art_mask = np.asarray(np.squeeze(read_mhd_image(art_mask_path)) == 1, np.uint8)
    pv_mask = np.asarray(np.squeeze(read_mhd_image(pv_mask_path)) == 1, np.uint8)

    height_nc, width_nc, perimeter_nc, area_nc, circle_metric_nc, edge_nc = calculate_mask_attributes(nc_mask)
    with open(os.path.join(cur_save_dir, 'NC_attributes.txt'), 'w') as nc_attr_file:
        line = '%d %d %d %d %.6f\n' % (height_nc, width_nc, perimeter_nc, area_nc, circle_metric_nc)
        nc_attr_file.writelines([line])
        nc_attr_file.close()

    height_art, width_art, perimeter_art, area_art, circle_metric_art, edge_art = calculate_mask_attributes(art_mask)
    with open(os.path.join(cur_save_dir, 'ART_attributes.txt'), 'w') as art_attr_file:
        line = '%d %d %d %d %.6f\n' % (height_art, width_art, perimeter_art, area_art, circle_metric_art)
        art_attr_file.writelines([line])
        art_attr_file.close()
    height_pv, width_pv, perimeter_pv, area_pv, circle_metric_pv, edge_pv = calculate_mask_attributes(pv_mask)
    with open(os.path.join(cur_save_dir, 'PV_attributes.txt'), 'w') as pv_attr_file:
        line = '%d %d %d %d %.6f\n' % (height_pv, width_pv, perimeter_pv, area_pv, circle_metric_pv)
        pv_attr_file.writelines([line])
        pv_attr_file.close()

    # print('edge_nc: ', np.min(edge_nc), np.max(edge_nc), np.unique(edge_nc))
    # print(height_nc, width_nc, perimeter_nc, area_nc, circle_metric_nc, cur_save_dir[-1])
    # cv2.imwrite(os.path.join(cur_save_dir, 'NC_EDGE.jpg'), edge_nc)

    if patient_name in window_center_dict.keys():
        cur_window_center = window_center_dict[patient_name]
    else:
        cur_window_center = [-70, 180, -70, 180, -70, 180]

    # print('cur_window_center is ', cur_window_center)
    nc_image = preprocessing_mhd(nc_image, [cur_window_center[0], cur_window_center[1]])
    art_image = preprocessing_mhd(art_image, [cur_window_center[2], cur_window_center[3]])
    pv_image = preprocessing_mhd(pv_image, [cur_window_center[4], cur_window_center[5]])

    nc_min_xs, nc_min_ys, nc_max_xs, nc_max_ys = bounding_box(nc_mask)
    art_min_xs, art_min_ys, art_max_xs, art_max_ys = bounding_box(art_mask)
    pv_min_xs, pv_min_ys, pv_max_xs, pv_max_ys = bounding_box(pv_mask)

    nc_roi = nc_image[nc_min_xs: nc_max_xs, nc_min_ys: nc_max_ys]
    art_roi = art_image[art_min_xs: art_max_xs, art_min_ys: art_max_ys]
    pv_roi = pv_image[pv_min_xs: pv_max_xs, pv_min_ys: pv_max_ys]

    nc_liver = nc_image[nc_liver_min_xs: nc_liver_max_xs, nc_liver_min_ys: nc_liver_max_ys]
    art_liver = art_image[art_liver_min_xs: art_liver_max_xs, art_liver_min_ys: art_liver_max_ys]
    pv_liver = pv_image[pv_liver_min_xs: pv_liver_max_xs, pv_liver_min_ys: pv_liver_max_ys]
    nc_mean_liver = np.mean(nc_liver)
    art_mean_liver = np.mean(art_liver)
    pv_mean_liver = np.mean(pv_liver)
    compared_nc_roi = nc_roi - nc_mean_liver
    compared_art_roi = art_roi - art_mean_liver
    compared_pv_roi = pv_roi - pv_mean_liver
    compared_min = np.min([np.min(compared_nc_roi), np.min(compared_art_roi), np.min(compared_pv_roi)])
    compared_max = np.max([np.max(compared_nc_roi), np.max(compared_art_roi), np.max(compared_pv_roi)])
    compared_nc_roi = (compared_nc_roi - compared_min) / (compared_max - compared_min)
    compared_art_roi = (compared_art_roi - compared_min) / (compared_max - compared_min)
    compared_pv_roi = (compared_pv_roi - compared_min) / (compared_max - compared_min)
    compared_nc_roi *= 255.0
    compared_art_roi *= 255.0
    compared_pv_roi *= 255.0
    # compared_nc_roi = np.asarray(compared_nc_roi * 255.0, np.uint8)
    # compared_art_roi = np.asarray(compared_art_roi * 255.0, np.uint8)
    # compared_pv_roi = np.asarray(compared_pv_roi * 255.0, np.uint8)

    # nc_roi = np.concatenate(
    #     [np.expand_dims(nc_roi, axis=2), np.expand_dims(nc_roi, axis=2), np.expand_dims(nc_roi, axis=2)], axis=2)
    # art_roi = np.concatenate(
    #     [np.expand_dims(art_roi, axis=2), np.expand_dims(art_roi, axis=2), np.expand_dims(art_roi, axis=2)], axis=2)
    # pv_roi = np.concatenate(
    #     [np.expand_dims(pv_roi, axis=2), np.expand_dims(pv_roi, axis=2), np.expand_dims(pv_roi, axis=2)], axis=2)
    # 将所有的病灶resize到最大的那个病灶上面
    nc_h, nc_w = np.shape(nc_roi)
    art_h, art_w = np.shape(art_roi)
    pv_h, pv_w = np.shape(pv_roi)
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
    nc_roi_resized = cv2.resize(nc_roi, (target_h, target_w))
    art_roi_resized = cv2.resize(art_roi, (target_h, target_w))
    pv_roi_resized = cv2.resize(pv_roi, (target_h, target_w))
    nc_liver_resized = cv2.resize(nc_liver, (target_h, target_w))
    art_liver_resized = cv2.resize(art_liver, (target_h, target_w))
    pv_liver_resized = cv2.resize(pv_liver, (target_h, target_w))

    nc_roi_compared_resized = np.asarray(cv2.resize(compared_nc_roi, (target_h, target_w)), np.uint8)
    art_roi_compared_resized = np.asarray(cv2.resize(compared_art_roi, (target_h, target_w)), np.uint8)
    pv_roi_compared_resized = np.asarray(cv2.resize(compared_pv_roi, (target_h, target_w)), np.uint8)

    nc_roi_final = np.concatenate([np.expand_dims(nc_roi_resized, axis=2), np.expand_dims(nc_liver_resized, axis=2),
                                   np.expand_dims(nc_roi_compared_resized, axis=2)], axis=2)
    art_roi_final = np.concatenate([np.expand_dims(art_roi_resized, axis=2), np.expand_dims(art_liver_resized, axis=2),
                                    np.expand_dims(art_roi_compared_resized, axis=2)], axis=2)
    pv_roi_final = np.concatenate([np.expand_dims(pv_roi_resized, axis=2), np.expand_dims(pv_liver_resized, axis=2),
                                   np.expand_dims(pv_roi_compared_resized, axis=2)], axis=2)

    patch_size = 5
    nc_patches, art_patches, pv_patches = extract_patches(nc_roi_final, art_roi_final, pv_roi_final, patch_size)

    nc_roi_save_path = os.path.join(cur_save_dir, 'NC.jpg')
    art_roi_save_path = os.path.join(cur_save_dir, 'ART.jpg')
    pv_roi_save_path = os.path.join(cur_save_dir, 'PV.jpg')
    cv2.imwrite(nc_roi_save_path, nc_roi_final)
    cv2.imwrite(art_roi_save_path, art_roi_final)
    cv2.imwrite(pv_roi_save_path, pv_roi_final)
    patch_dir = os.path.join(cur_save_dir, '%dx%d' % (patch_size, patch_size))
    if not os.path.exists(patch_dir):
        os.mkdir(patch_dir)
    for idx, (nc_patch, art_patch, pv_patch) in enumerate(zip(nc_patches, art_patches, pv_patches)):
        nc_patch_save_path = os.path.join(patch_dir, '%d_NC.jpg' % idx)
        art_patch_save_path = os.path.join(patch_dir, '%d_ART.jpg' % idx)
        pv_patch_save_path = os.path.join(patch_dir, '%d_PV.jpg' % idx)
        cv2.imwrite(nc_patch_save_path, nc_patch)
        cv2.imwrite(art_patch_save_path, art_patch)
        cv2.imwrite(pv_patch_save_path, pv_patch)
    # print(slice_dir, 'num of patch is ', len(nc_patches))


def LiverLesionDetection_Iterator(image_dir, execute_func, *parameters):
    '''
    遍历MICCAI2018文件夹的框架
    :param execute_func:
    :return:
    '''
    for sub_name in ['train', 'val', 'test']:
        names = os.listdir(os.path.join(image_dir, sub_name))
        for name in names:
            cur_slice_dir = os.path.join(image_dir, sub_name, name)
            execute_func(cur_slice_dir, *parameters)


def extract_bboxs_from_mask(mask_image, tumor_types):
    mask_image = mask_image[1, :, :]
    w, h = np.shape(mask_image)
    if w != 512 or h != 512:
        print(np.shape(mask_image))
        assert False
    with open(tumor_types, 'r') as f:
        lines = f.readlines()
        idx2names = {}
        for line in lines:
            line = line[:-1]
            idx, name = line.split(' ')
            idx2names[idx] = name

        maximum = np.max(mask_image)
        min_xs = []
        min_ys = []
        max_xs = []
        max_ys = []
        names = []
        masks = []
        for i in range(1, maximum + 1):
            cur_mask_image = np.asarray(mask_image == i, np.uint8)
            masks.append(cur_mask_image)
            if np.sum(cur_mask_image) == 0:
                continue
            xs, ys = np.where(cur_mask_image == 1)
            min_x = np.min(xs)
            min_y = np.min(ys)
            max_x = np.max(xs)
            max_y = np.max(ys)
            min_xs.append(min_x)
            min_ys.append(min_y)
            max_xs.append(max_x)
            max_ys.append(max_y)
            names.append(idx2names[str(i)])
    return min_xs, min_ys, max_xs, max_ys, names, masks


def check_multiphase_tripleslice(slice_dir, save_dir, phase_names, target_phase_name):
    def is_eauql(list_1, list_2):
        if len(list_1) != len(list_2):
            return False
        list_1 = sorted(list_1)
        list_2 = sorted(list_2)
        for ele_1, ele_2 in zip(list_1, list_2):
            if ele_1 != ele_2:
                return False
        return True
    total_phase_name = ''.join(phase_names)
    total_phase_name += '_tripleslice'

    mhd_images = []
    mask_images = []
    for phase_name in phase_names:
        mhd_image_path = os.path.join(slice_dir, 'Image_' + phase_name + '.mhd')
        mhd_mask_path = os.path.join(slice_dir, 'Mask_' + phase_name + '.mhd')
        mhd_image = read_mhd_image(mhd_image_path)
        mask_image = read_mhd_image(mhd_mask_path)
        mhd_image = np.asarray(np.squeeze(mhd_image), np.float32)
        mhd_image = np.transpose(mhd_image, axes=[1, 2, 0])

        _, _, depth_image = np.shape(mhd_image)
        if depth_image == 2:
            mhd_image = np.concatenate(
                [mhd_image,
                 np.expand_dims(mhd_image[:, :, np.argmax(np.sum(np.sum(mask_image, axis=1), axis=1))], axis=2)],
                axis=2
            )
            print('Error')
        mhd_images.append(mhd_image)
        mask_images.append(mask_image)
    nc_unique = np.unique(mask_images[0])
    art_unique = np.unique(mask_images[1])
    pv_unique = np.unique(mask_images[2])
    if is_eauql(nc_unique, art_unique) and is_eauql(art_unique, pv_unique):
        print('OK ', slice_dir)
    else:
        print('Error ', slice_dir, nc_unique, art_unique, pv_unique)


def dicom2jpg_multiphase_tripleslice(slice_dir, save_dir, phase_names, target_phase_name):
    '''
    前置条件：已经将dicom格式的数据转为成MHD格式，并且已经提出了slice，一个mhd文件只包含了三个slice
    针对每个phase提取三个slice(all), 但是mask还是只提取一个
    保存的格式是name_nc.jpg, name_art.jpg, name_pv.jpg
    :param slice_dir:
    :param save_dir:
    :param phase_names:
    :param target_phase_name:
    :return:
    '''
    total_phase_name = ''.join(phase_names)
    total_phase_name += '_tripleslice'
    target_phase_mask = None
    mhd_images = []
    for phase_name in phase_names:
        mhd_image_path = os.path.join(slice_dir, 'Image_' + phase_name + '.mhd')
        mhd_mask_path = os.path.join(slice_dir, 'Mask_' + phase_name + '.mhd')
        mhd_image = read_mhd_image(mhd_image_path)
        mask_image = read_mhd_image(mhd_mask_path)
        mhd_image = np.asarray(np.squeeze(mhd_image), np.float32)
        mhd_image = np.transpose(mhd_image, axes=[1, 2, 0])
        if phase_name == target_phase_name:
            target_phase_mask = mask_image
        _, _, depth_image = np.shape(mhd_image)
        if depth_image == 2:
            mhd_image = np.concatenate(
                [mhd_image,
                 np.expand_dims(mhd_image[:, :, np.argmax(np.sum(np.sum(mask_image, axis=1), axis=1))], axis=2)],
                axis=2
            )
            print('Error')
        mhd_images.append(mhd_image)
    # mhd_image = np.expand_dims(mhd_image, axis=2)
    # mhd_image = np.concatenate([mhd_image, mhd_image, mhd_image], axis=2)
    mhd_image = np.concatenate(mhd_images, axis=-1)
    mask_image = target_phase_mask
    mask_image = np.asarray(np.squeeze(mask_image), np.uint8)
    max_v = 300.
    min_v = -350.
    mhd_image[mhd_image > max_v] = max_v
    mhd_image[mhd_image < min_v] = min_v
    print(np.mean(mhd_image, dtype=np.float32))
    mhd_image -= np.mean(mhd_image)
    min_v = np.min(mhd_image)
    max_v = np.max(mhd_image)
    interv = max_v - min_v
    mhd_image = (mhd_image - min_v) / interv
    file_name = os.path.basename(slice_dir)
    dataset_name = os.path.basename(os.path.dirname(slice_dir))

    print('the shape of mhd_image is ', np.shape(mhd_image), np.min(mhd_image), np.max(mhd_image))
    print('the shape of mask_image is ', np.shape(mask_image))
    min_xs, min_ys, max_xs, max_ys, names, masks = extract_bboxs_from_mask(mask_image,
                                                                           os.path.join(slice_dir, 'tumor_types'))
    for mask, name in zip(masks, names):
        print name, np.shape(mask)


def convertMHD2NPY(dataset_dir, stage_name):
    def _get_filled_mask(mask):
        from medicalImage import fill_region
        mask = np.asarray(mask == 1, np.uint8)
        mask = fill_region(mask)
        xs, ys = np.where(mask == 1)
        return mask, np.min(xs), np.min(ys), np.max(xs), np.max(ys)

    stage_dir = os.path.join(dataset_dir, stage_name)
    slice_names = os.listdir(stage_dir)
    labels = []
    imgs = []
    output_shape = (227, 227)
    for slice_name in slice_names:
        if slice_name.startswith('.DS'):
            continue
        print slice_name
        slice_dir = os.path.join(stage_dir, slice_name)
        nc_img_path = glob(os.path.join(slice_dir, 'NC_Image*.mhd'))[0]
        art_img_path = glob(os.path.join(slice_dir, 'ART_Image*.mhd'))[0]
        pv_img_path = glob(os.path.join(slice_dir, 'PV_Image*.mhd'))[0]
        nc_img = np.squeeze(read_mhd_image(nc_img_path))
        art_img = np.squeeze(read_mhd_image(art_img_path))
        pv_img = np.squeeze(read_mhd_image(pv_img_path))
        nc_mask_path = glob(os.path.join(slice_dir, 'NC_Mask*.mhd'))[0]
        art_mask_path = glob(os.path.join(slice_dir, 'ART_Mask*.mhd'))[0]
        pv_mask_path = glob(os.path.join(slice_dir, 'PV_Mask*.mhd'))[0]
        nc_mask = np.squeeze(read_mhd_image(nc_mask_path))
        art_mask = np.squeeze(read_mhd_image(art_mask_path))
        pv_mask = np.squeeze(read_mhd_image(pv_mask_path))

        nc_mask, nc_min_xs, nc_min_ys, nc_max_xs, nc_max_ys = _get_filled_mask(nc_mask)
        art_mask, art_min_xs, art_min_ys, art_max_xs, art_max_ys = _get_filled_mask(art_mask)
        pv_mask, pv_min_xs, pv_min_ys, pv_max_xs, pv_max_ys = _get_filled_mask(pv_mask)

        nc_img[nc_mask != 1] = 0
        art_img[art_mask != 1] = 0
        pv_img[pv_mask != 1] = 0
        nc_img = nc_img[nc_min_xs: nc_max_xs, nc_min_ys: nc_max_ys]
        art_img = art_img[art_min_xs: art_max_xs, art_min_ys: art_max_ys]
        pv_img = pv_img[pv_min_xs: pv_max_xs, pv_min_ys: pv_max_ys]

        nc_img = np.asarray(nc_img, np.float32)
        art_img = np.asarray(art_img, np.float32)
        pv_img = np.asarray(pv_img, np.float32)

        nc_img = cv2.resize(nc_img, output_shape)
        art_img = cv2.resize(art_img, output_shape)
        pv_img = cv2.resize(pv_img, output_shape)
        img = np.concatenate([np.expand_dims(nc_img, axis=2), np.expand_dims(art_img, axis=2),
                              np.expand_dims(pv_img, axis=2)], axis=2)
        imgs.append(img)
        labels.append(int(os.path.basename(slice_dir)[-1]))
    imgs = np.asarray(imgs)
    labels = np.asarray(labels)
    print np.shape(imgs), np.shape(labels)
    np.save(os.path.join(dataset_dir, stage_name+'_imgs.npy'), imgs)
    np.save(os.path.join(dataset_dir, stage_name+'_labels.npy'), labels)


def convertJPG2NPY(dataset_dir, stage_name):
    stage_dir = os.path.join(dataset_dir, stage_name)
    slice_names = os.listdir(stage_dir)
    labels = []
    imgs = []
    output_shape = (224, 224)
    for slice_name in slice_names:
        slice_dir = os.path.join(stage_dir, slice_name)
        labels.append(int(os.path.basename(slice_dir)[-1]))
        nc_img_path = os.path.join(slice_dir, 'NC.jpg')
        art_img_path = os.path.join(slice_dir, 'ART.jpg')
        pv_img_path = os.path.join(slice_dir, 'PV.jpg')
        nc_img = cv2.imread(nc_img_path)
        art_img = cv2.imread(art_img_path)
        pv_img = cv2.imread(pv_img_path)
        nc_img = cv2.resize(nc_img, output_shape)
        art_img = cv2.resize(art_img, output_shape)
        pv_img = cv2.resize(pv_img, output_shape)
        img = np.concatenate([np.expand_dims(nc_img[:, :, 1], axis=2), np.expand_dims(art_img[:, :, 1], axis=2),
                              np.expand_dims(pv_img[:, :, 1], axis=2)], axis=2)
        imgs.append(img)
    imgs = np.asarray(imgs)
    labels = np.asarray(labels)
    print np.shape(imgs), np.shape(labels)
    np.save(os.path.join(dataset_dir, stage_name+'_imgs.npy'), imgs)
    np.save(os.path.join(dataset_dir, stage_name+'_labels.npy'), labels)


if __name__ == '__main__':
    # image_dir = '/home/dl-box/ld/Documents/datasets/IEEEonMedicalImage_Splited/0'
    # LiverLesionDetection_Iterator(
    #     image_dir,
    #     check_multiphase_tripleslice,
    #     '/home/dl-box/ld/Documents/datasets/IEEEonMedicalImage_Splited/JPG/0',
    #     ['NC', 'ART', 'PV'],
    #     'PV'
    # )

    image_dir = '/home/dl-box/ld/Documents/datasets/IEEEonMedicalImage_Splited/0'
    MICCAI2018_Iterator(
        image_dir,
        convert2jpg_multiphase_with_liver,
        '/home/dl-box/ld/Documents/datasets/IEEEonMedicalImage_Splited/JPG/0_attribute_liver'
    )

    # for stage_name in ['train', 'val', 'test']:
    #     convertJPG2NPY('/home/dl-box/ld/Documents/datasets/IEEEonMedicalImage_Splited/JPG/0_attribute_liver',
    #                    stage_name)
    # convertMHD2NPY('/media/dl-box/HDCZ-UT/datasets/MICCAI2018/Slices/crossvalidation/0', 'train')
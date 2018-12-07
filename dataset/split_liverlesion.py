# -*- coding=utf-8 -*-
'''
完成对liver lesion detection文件夹的划分
'''
import os
import numpy as np
import shutil
from glob import glob


def splited(liver_lesion_detection_dir, saved_dir):
    names = os.listdir(liver_lesion_detection_dir)
    iid_sets = []
    for name in names:
        if not os.path.isdir(os.path.join(liver_lesion_detection_dir, name)):
            continue
        eles = name.split('-')
        iid_sets.append('-'.join(eles[:3]))
    print(iid_sets)
    print(len(iid_sets))
    iid_sets = list(set(iid_sets))
    print(iid_sets)
    print(len(iid_sets))
    for iid_name in iid_sets:
        random_v = np.random.random()
        if random_v < 0.6:
            # mv to train:
            final_saved_dir = os.path.join(saved_dir, 'train')
            print(final_saved_dir)
            # iid_names = glob(os.path.join(liver_lesion_detection_dir, iid_name + '*'))
            # [shutil.copytree(os.path.join(liver_lesion_detection_dir, iid_name_ele),
            #                  os.path.join(final_saved_dir, iid_name_ele)) for iid_name_ele in iid_names]
        elif random_v < 0.8:
            final_saved_dir = os.path.join(saved_dir, 'val')
            print(final_saved_dir)
            # iid_names = glob(os.path.join(liver_lesion_detection_dir, iid_name + '*'))
            # [shutil.copytree(os.path.join(liver_lesion_detection_dir, iid_name_ele),
            #                  os.path.join(final_saved_dir, iid_name_ele)) for iid_name_ele in iid_names]
        else:
            final_saved_dir = os.path.join(saved_dir, 'test')
            print(final_saved_dir)
            # iid_names = glob(os.path.join(liver_lesion_detection_dir, iid_name + '*'))
            # [shutil.copytree(os.path.join(liver_lesion_detection_dir, iid_name_ele),
            #                  os.path.join(final_saved_dir, iid_name_ele)) for iid_name_ele in iid_names]
        iid_names = glob(os.path.join(liver_lesion_detection_dir, iid_name + '-*'))
        print(iid_name)
        for iid_name_ele in iid_names:
            iid_name_ele = os.path.basename(iid_name_ele)
            print('final_saved_dir: ', final_saved_dir)
            print('iid_name_ele: ', iid_name_ele)
            print('src: ', os.path.join(liver_lesion_detection_dir, iid_name_ele))
            print('dst: ', os.path.join(final_saved_dir, iid_name_ele))
            shutil.copytree(os.path.join(liver_lesion_detection_dir, iid_name_ele),
                            os.path.join(final_saved_dir, iid_name_ele))


if __name__ == '__main__':
    splited(
        '/home/dl-box/ld/Documents/datasets/IEEEonMedicalImage',
        '/home/dl-box/ld/Documents/datasets/IEEEonMedicalImage_Splited/0'
    )
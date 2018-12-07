import numpy as np
from Tools import read_mhd_image, get_boundingbox, convert2depthlaster
import os
from glob import glob
from PIL import Image


def visulization(data_dir):
    img = None
    for phasename in ['NC', 'ART', 'PV']:
        mhd_path = glob(os.path.join(data_dir, phasename+'_Image*.mhd'))[0]
        mask_path = glob(os.path.join(data_dir, phasename+'_Registration*.mhd'))[0]
        mhd_image = read_mhd_image(mhd_path)
        mask_image = read_mhd_image(mask_path)
        mhd_image = np.squeeze(mhd_image)
        mask_image = np.squeeze(mask_image)
        x_min, x_max, y_min, y_max = get_boundingbox(mask_image)
        ROI = mhd_image[x_min: x_max, y_min: y_max]
        print np.shape(ROI)
        if img is None:
            img = []
        img.append(ROI)
    img = convert2depthlaster(img)
    print np.shape(img)
    img = Image.fromarray(np.asarray(img, np.uint8))
    img.show()
    img.save('./multi-phase_ROI.png')


if __name__ == '__main__':
    data_dir = '/home/give/Documents/dataset/MICCAI2018/Slices/crossvalidation/0/test/2965287_2809549_0_0_2'
    visulization(data_dir)
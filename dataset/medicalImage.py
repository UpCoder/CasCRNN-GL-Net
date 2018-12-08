# -*- coding=utf-8 -*-
import SimpleITK as itk
import dicom as pydicom
import numpy as np
from PIL import Image, ImageDraw
import gc
from skimage.morphology import disk, dilation
import nipy
import os
from glob import glob
import scipy
import cv2
from xml.dom.minidom import Document

typenames = ['CYST', 'FNH', 'HCC', 'HEM', 'METS']
typeids = [0, 1, 2, 3, 4]


def get_voxel_size(file_path):
    load_image_obj = nipy.load_image(file_path)
    header = load_image_obj.header
    x_size = header['srow_x'][0]
    y_size = header['srow_y'][1]
    z_size = header['srow_z'][2]
    return [x_size, y_size, z_size]


def read_nil(file_path):
    return nipy.load_image(file_path).get_data()


# 读取文件序列
def read_dicom_series(dir_name):
    reader = itk.ImageSeriesReader()
    dicom_series = reader.GetGDCMSeriesFileNames(dir_name)
    reader.SetFileNames(dicom_series)
    images = reader.Execute()
    image_array = itk.GetArrayFromImage(images)
    return image_array


# 将DICOM序列转化成MHD文件
def convert_dicomseries2mhd(dicom_series_dir, save_path):
    data = read_dicom_series(dicom_series_dir)
    save_mhd_image(data, save_path)


# 读取单个DICOM文件
def read_dicom_file(file_name):
    header = pydicom.read_file(file_name)
    image = header.pixel_array
    image = header.RescaleSlope * image + header.RescaleIntercept
    return image



# 读取mhd文件
def read_mhd_image(file_path, rejust=False):
    header = itk.ReadImage(file_path)
    image = np.array(itk.GetArrayFromImage(header))
    if rejust:
        image[image < -70] = -70
        image[image > 180] = 180
        image = image + 70
    return np.array(image)


# 保存mhd文件
def save_mhd_image(image, file_name):
    header = itk.GetImageFromArray(image)
    itk.WriteImage(header, file_name)


# 根据文件名返回期项名
def return_phasename(file_name):
    phasenames = ['NC', 'ART', 'PV']
    for phasename in phasenames:
        if file_name.find(phasename) != -1:
            return phasename


# 读取DICOM文件中包含的病例ID信息
def read_patientId(dicom_file_path):
    ds = pydicom.read_file(dicom_file_path)
    return ds.PatientID


# 返回病灶类型和ID的字典类型的数据 key是typename value是typeid
def return_type_nameid():
    res = {}
    res['CYST'] = 0
    res['FNH'] = 1
    res['HCC'] = 2
    res['HEM'] = 3
    res['METS'] = 4
    return res


# 返回病灶类型ID和名称的字典类型的数据 key是typeid value是typename
def return_type_idname():
    res = {}
    res[0] = 'CYST'
    res[1] = 'FNH'
    res[2] = 'HCC'
    res[3] = 'HEM'
    res[4] = 'METS'
    return res


# 根据病灶类型的ID返回类型的字符串
def return_typename_byid(typeid):
    idname_dict = return_type_idname()
    return idname_dict[typeid]


# 根据病灶类型的name返回id的字符串
def return_typeid_byname(typename):
    nameid_dict = return_type_nameid()
    return nameid_dict[typename]


# 填充图像
def fill_region(image):
    # image.show()
    from scipy import ndimage
    image = ndimage.binary_fill_holes(image).astype(np.uint8)
    return image


def open_operation(slice_image, kernel_size=3):
    opening = cv2.morphologyEx(slice_image, cv2.MORPH_OPEN,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)))
    return opening
# 图像膨胀
# def image_expand(image, size):
#    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
#    image = cv2.dilate(image, kernel)
#    return image

def find_significant_layer(mask_image):
    '''
    找到显著层
    :param mask_image: [depth, width, height]
    :return: idx
    '''
    sum_res = np.sum(np.sum(mask_image, axis=1), axis=1)
    return np.argmax(sum_res)


# 将一个矩阵保存为图片
def save_image(image_arr, save_path):
    image = Image.fromarray(np.asarray(image_arr, np.uint8))
    image.save(save_path)


def show_image(image):
    img = Image.fromarray(np.asarray(image, np.uint8))
    img.show()


# 将图像画出来，并且画出标记的病灶
def save_image_with_mask(image_arr, mask_image, save_path):
    image_arr[image_arr < -70] = -70
    image_arr[image_arr > 180] = 180
    image_arr = image_arr + 70
    shape = list(np.shape(image_arr))
    image_arr_rgb = np.zeros(shape=[shape[0], shape[1], 3])
    image_arr_rgb[:, :, 0] = image_arr
    image_arr_rgb[:, :, 1] = image_arr
    image_arr_rgb[:, :, 2] = image_arr
    image = Image.fromarray(np.asarray(image_arr_rgb, np.uint8))
    image_draw = ImageDraw.Draw(image)
    [ys, xs] = np.where(mask_image != 0)
    miny = np.min(ys)
    maxy = np.max(ys)
    minx = np.min(xs)
    maxx = np.max(xs)
    ROI = image_arr_rgb[miny - 1:maxy + 1, minx - 1:maxx + 1, :]
    ROI_Image = Image.fromarray(np.asarray(ROI, np.uint8))

    for index, y in enumerate(ys):
        image_draw.point([xs[index], y], fill=(255, 0, 0))
    if save_path is None:
        image.show()
    else:
        image.save(save_path)
        ROI_Image.save(os.path.join(os.path.dirname(save_path), os.path.basename(save_path).split('.')[0] + '_ROI.jpg'))
        del image, ROI_Image
        gc.collect()


def compress22dim(image):
    '''
        将一个矩阵如果可能，压缩到三维的空间
    '''
    shape = list(np.shape(image))
    if len(shape) == 3:
        return np.squeeze(image)
    return image


def extract_ROI(image, mask_image):
    '''
        提取一幅图像中的ＲＯＩ
    '''
    xs, ys = np.where(mask_image == 1)
    xs_min = np.min(xs)
    xs_max = np.max(xs)
    ys_min = np.min(ys)
    ys_max = np.max(ys)
    return image[xs_min: xs_max + 1, ys_min: ys_max + 1]


def resize_image(image, size):
    image = Image.fromarray(np.asarray(image, np.uint8))
    return image.resize((size, size))


def image_expand(mask_image, r):
    return dilation(mask_image, disk(r))


'''
    将形式如(512, 512)格式的图像转化为(1, 512, 512)形式的图片
'''
def expand23D(mask_image):
    shape = list(np.shape(mask_image))
    if len(shape) == 2:
        mask_image = np.expand_dims(mask_image, axis=0)
        print('after expand23D', np.shape(mask_image))
    return mask_image


'''
    返回一个ｍａｓｋ图像的中心，是对ｘｙｚ坐标计算平均值之后的结果
'''
def find_centroid3D(image, flag):
    [x, y, z] = np.where(image == flag)
    centroid_x = int(np.mean(x))
    centroid_y = int(np.mean(y))
    centroid_z = int(np.mean(z))
    return centroid_x, centroid_y, centroid_z


'''
    将[w, h, d]reshape为[d, w, h]
'''
def convert2depthfirst(image):
    image = np.array(image)
    shape = np.shape(image)
    new_image = np.zeros([shape[2], shape[0], shape[1]])
    for i in range(shape[2]):
        new_image[i, :, :] = image[:, :, i]
    return new_image
    # def test_convert2depthfirst():
    #     zeros = np.zeros([100, 100, 30])
    #     after_zeros = convert2depthfirst(zeros)
    #     print np.shape(after_zeros)
    # test_convert2depthfirst()

'''
    将[d, w, h]reshape为[w, h, d]
'''
def convert2depthlastest(image):
    image = np.array(image)
    shape = np.shape(image)
    new_image = np.zeros([shape[1], shape[2], shape[0]])
    for i in range(shape[0]):
        new_image[:, :, i] = image[i, :, :]
    return new_image


def read_image_file(file_path):
    if file_path.endswith('.nii'):
        return read_nil(file_path)
    if file_path.endswith('.mhd'):
        return read_mhd_image(file_path)
    print('the format of image is not support in this version')
    return None


def processing(image, size_training):
    image = np.array(image)
    # numpy_clip
    bottom = -300.
    top = 500.
    image = np.clip(image, bottom, top)

    # to float
    minval = -350
    interv = 500 - (-350)
    image -= minval
    # scale down to 0 - 2
    image /= (interv / 2)

    # zoom
    desired_size = [size_training, size_training]
    desired_size = np.asarray(desired_size, dtype=np.int)
    zooms = desired_size / np.array(image[:, :, 0].shape, dtype=np.float)
    print(zooms)
    after_zoom = np.zeros([size_training, size_training, np.shape(image)[2]])
    for i in range(np.shape(after_zoom)[2]):
        after_zoom[:, :, i] = scipy.ndimage.zoom(image[:, :, i], zooms, order=1)  # order = 1 => biliniear interpolation

    return after_zoom


def preprocessing_agumentation(image, size_training):
    image = np.array(image)
    # numpy_clip
    c_minimum = -300.
    c_maximum = 500.
    s_maximum = 255.
    image = np.clip(image, c_minimum, c_maximum)

    interv = float(c_maximum - c_minimum)
    image = (image - c_minimum) / interv * s_maximum
    minval = 0.
    maxval = 255.
    image -= minval
    interv = maxval - minval
    # print('static scaler 0', interv)
    # scale down to 0 - 2
    # image /= (interv / 2)
    image = np.asarray(image, np.float32)
    image = image / interv
    image = image * 2.0
    # zoom
    desired_size = [size_training, size_training]
    desired_size = np.asarray(desired_size, dtype=np.int)
    zooms = desired_size / np.array(image[:, :, 0].shape, dtype=np.float)
    print(zooms)
    after_zoom = np.zeros([size_training, size_training, np.shape(image)[2]])
    for i in range(np.shape(after_zoom)[2]):
        after_zoom[:, :, i] = scipy.ndimage.zoom(image[:, :, i], zooms, order=1)  # order = 1 => biliniear interpolation

    return after_zoom

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
            execute_func(cur_slice_dir, *parameters)


def dicom2jpg_singlephase(slice_dir, save_dir, phase_name='PV'):
    mhd_image_path = glob(os.path.join(slice_dir, phase_name+'_Image*.mhd'))[0]
    mhd_mask_path = glob(os.path.join(slice_dir, phase_name + '_Mask*.mhd'))[0]
    mhd_image = read_mhd_image(mhd_image_path)
    mask_image = read_mhd_image(mhd_mask_path)
    mhd_image = np.asarray(np.squeeze(mhd_image), np.float32)
    mhd_image = np.expand_dims(mhd_image, axis=2)
    mhd_image = np.concatenate([mhd_image, mhd_image, mhd_image], axis=2)
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
    save_path = os.path.join(save_dir, phase_name, dataset_name, file_name+'.jpg')
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    print('the shape of mhd_image is ', np.shape(mhd_image), np.min(mhd_image), np.max(mhd_image))
    cv2.imwrite(save_path, mhd_image * 255)

    xml_save_dir = os.path.join(save_dir, phase_name, dataset_name+'_xml')
    if not os.path.exists(xml_save_dir):
        os.makedirs(xml_save_dir)

    evulate_gt_dir = os.path.join(save_dir, phase_name, dataset_name+'_gt')
    if not os.path.exists(evulate_gt_dir):
        os.makedirs(evulate_gt_dir)

    xml_save_path  = os.path.join(xml_save_dir, file_name + '.xml')
    gt_save_path = os.path.join(evulate_gt_dir, file_name + '.txt') # for evulate

    doc = Document()
    root_node = doc.createElement('annotation')
    doc.appendChild(root_node)

    folder_name = os.path.basename(save_dir) + '/' + phase_name
    folder_node = doc.createElement('folder')
    root_node.appendChild(folder_node)
    folder_txt_node = doc.createTextNode(folder_name)
    folder_node.appendChild(folder_txt_node)

    file_name = file_name + '.jpg'
    filename_node = doc.createElement('filename')
    root_node.appendChild(filename_node)
    filename_txt_node = doc.createTextNode(file_name)
    filename_node.appendChild(filename_txt_node)

    shape = list(np.shape(mhd_image))
    size_node = doc.createElement('size')
    root_node.appendChild(size_node)
    width_node = doc.createElement('width')
    width_node.appendChild(doc.createTextNode(str(shape[0])))
    height_node = doc.createElement('height')
    height_node.appendChild(doc.createTextNode(str(shape[1])))
    depth_node = doc.createElement('depth')
    depth_node.appendChild(doc.createTextNode(str(3)))
    size_node.appendChild(width_node)
    size_node.appendChild(height_node)
    size_node.appendChild(depth_node)

    mask_image[mask_image != 1] = 0
    xs, ys = np.where(mask_image == 1)
    min_x = np.min(xs)
    min_y = np.min(ys)
    max_x = np.max(xs)
    max_y = np.max(ys)
    object_node = doc.createElement('object')
    root_node.appendChild(object_node)
    name_node = doc.createElement('name')
    name_node.appendChild(doc.createTextNode('Cyst'))
    object_node.appendChild(name_node)
    truncated_node = doc.createElement('truncated')
    object_node.appendChild(truncated_node)
    truncated_node.appendChild(doc.createTextNode('0'))
    difficult_node = doc.createElement('difficult')
    object_node.appendChild(difficult_node)
    difficult_node.appendChild(doc.createTextNode('0'))

    bndbox_node = doc.createElement('bndbox')
    object_node.appendChild(bndbox_node)
    xmin_node = doc.createElement('xmin')
    xmin_node.appendChild(doc.createTextNode(str(min_y)))
    bndbox_node.appendChild(xmin_node)

    ymin_node = doc.createElement('ymin')
    ymin_node.appendChild(doc.createTextNode(str(min_x)))
    bndbox_node.appendChild(ymin_node)

    xmax_node = doc.createElement('xmax')
    xmax_node.appendChild(doc.createTextNode(str(max_y)))
    bndbox_node.appendChild(xmax_node)

    ymax_node = doc.createElement('ymax')
    ymax_node.appendChild(doc.createTextNode(str(max_x)))
    bndbox_node.appendChild(ymax_node)
    with open(xml_save_path, 'wb') as f:
        f.write(doc.toprettyxml(indent='\t', encoding='utf-8'))

    line = '%s %d %d %d %d\n' % ('Cyst', min_y, min_x, max_y, max_x)
    print(line)
    lines = []
    lines.append(line)
    with open(gt_save_path, 'w') as f:
        f.writelines(lines)
        f.close()

def dicom2jpg_multiphase(slice_dir, save_dir, phasenames=['NC', 'ART', 'PV'], target_phase='PV'):
    target_mask = None
    mhd_images = []
    for phase_name in phasenames:
        mhd_image_path = glob(os.path.join(slice_dir, phase_name+'_Image*.mhd'))[0]
        mhd_mask_path = glob(os.path.join(slice_dir, phase_name + '_Mask*.mhd'))[0]
        mhd_image = read_mhd_image(mhd_image_path)
        mask_image = read_mhd_image(mhd_mask_path)
        mhd_image = np.asarray(np.squeeze(mhd_image), np.float32)
        mhd_images.append(mhd_image)
        mask_image = np.asarray(np.squeeze(mask_image), np.uint8)
        if phase_name == target_phase:
            target_mask = mask_image
    print(np.shape(mhd_images))
    mhd_images = np.transpose(np.asarray(mhd_images, np.float32), axes=[1, 2, 0])
    mhd_image = mhd_images
    mask_image = target_mask

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
    phase_name = ''.join(phasenames)
    save_path = os.path.join(save_dir, phase_name, dataset_name, file_name+'.jpg')
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    print('the shape of mhd_image is ', np.shape(mhd_image), np.min(mhd_image), np.max(mhd_image))
    cv2.imwrite(save_path, mhd_image * 255)

    xml_save_dir = os.path.join(save_dir, phase_name, dataset_name+'_xml')
    if not os.path.exists(xml_save_dir):
        os.makedirs(xml_save_dir)

    evulate_gt_dir = os.path.join(save_dir, phase_name, dataset_name+'_gt')
    if not os.path.exists(evulate_gt_dir):
        os.makedirs(evulate_gt_dir)

    xml_save_path  = os.path.join(xml_save_dir, file_name + '.xml')
    gt_save_path = os.path.join(evulate_gt_dir, file_name + '.txt') # for evulate

    doc = Document()
    root_node = doc.createElement('annotation')
    doc.appendChild(root_node)

    folder_name = os.path.basename(save_dir) + '/' + phase_name
    folder_node = doc.createElement('folder')
    root_node.appendChild(folder_node)
    folder_txt_node = doc.createTextNode(folder_name)
    folder_node.appendChild(folder_txt_node)

    file_name = file_name + '.jpg'
    filename_node = doc.createElement('filename')
    root_node.appendChild(filename_node)
    filename_txt_node = doc.createTextNode(file_name)
    filename_node.appendChild(filename_txt_node)

    shape = list(np.shape(mhd_image))
    size_node = doc.createElement('size')
    root_node.appendChild(size_node)
    width_node = doc.createElement('width')
    width_node.appendChild(doc.createTextNode(str(shape[0])))
    height_node = doc.createElement('height')
    height_node.appendChild(doc.createTextNode(str(shape[1])))
    depth_node = doc.createElement('depth')
    depth_node.appendChild(doc.createTextNode(str(3)))
    size_node.appendChild(width_node)
    size_node.appendChild(height_node)
    size_node.appendChild(depth_node)

    mask_image[mask_image != 1] = 0
    xs, ys = np.where(mask_image == 1)
    min_x = np.min(xs)
    min_y = np.min(ys)
    max_x = np.max(xs)
    max_y = np.max(ys)
    object_node = doc.createElement('object')
    root_node.appendChild(object_node)
    name_node = doc.createElement('name')
    name_node.appendChild(doc.createTextNode('Cyst'))
    object_node.appendChild(name_node)
    truncated_node = doc.createElement('truncated')
    object_node.appendChild(truncated_node)
    truncated_node.appendChild(doc.createTextNode('0'))
    difficult_node = doc.createElement('difficult')
    object_node.appendChild(difficult_node)
    difficult_node.appendChild(doc.createTextNode('0'))

    bndbox_node = doc.createElement('bndbox')
    object_node.appendChild(bndbox_node)
    xmin_node = doc.createElement('xmin')
    xmin_node.appendChild(doc.createTextNode(str(min_y)))
    bndbox_node.appendChild(xmin_node)

    ymin_node = doc.createElement('ymin')
    ymin_node.appendChild(doc.createTextNode(str(min_x)))
    bndbox_node.appendChild(ymin_node)

    xmax_node = doc.createElement('xmax')
    xmax_node.appendChild(doc.createTextNode(str(max_y)))
    bndbox_node.appendChild(xmax_node)

    ymax_node = doc.createElement('ymax')
    ymax_node.appendChild(doc.createTextNode(str(max_x)))
    bndbox_node.appendChild(ymax_node)
    with open(xml_save_path, 'wb') as f:
        f.write(doc.toprettyxml(indent='\t', encoding='utf-8'))

    line = '%s %d %d %d %d\n' % ('Cyst', min_y, min_x, max_y, max_x)
    print(line)
    lines = []
    lines.append(line)
    with open(gt_save_path, 'w') as f:
        f.writelines(lines)
        f.close()


def generate_paths(dataset_dir, stage_name, shuffling=True):
    import random
    cur_dataset_dir = os.path.join(dataset_dir, stage_name)
    slice_names = os.listdir(cur_dataset_dir)
    patch_size = 5
    RANDOM_SEED = 10
    nc_roi_paths = []
    art_roi_paths = []
    pv_roi_paths = []
    nc_patches_paths = []
    art_patches_paths = []
    pv_patches_paths = []
    labels = []
    for slice_name in slice_names:
        print(slice_name)
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
              len(pv_patches_paths))
    return nc_roi_paths, art_roi_paths, pv_roi_paths, nc_patches_paths, art_patches_paths, pv_patches_paths, labels


def calculate_mean_pixel():
    nc_roi_paths, art_roi_paths, pv_roi_paths, nc_patches_paths, art_patches_paths, pv_patches_paths, labels = generate_paths(
        '/home/dl-box/ld/Documents/datasets/IEEEonMedicalImage_Splited/JPG/0',
        'train', shuffling=True
    )
    mean_nc = np.asarray([0., 0., 0.])
    mean_art = np.asarray([0., 0., 0.])
    mean_pv = np.asarray([0., 0., 0.])
    class_statics = {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
        4: 0
    }
    class_statics[0] = np.sum(labels == 0)
    class_statics[1] = np.sum(labels == 1)
    class_statics[2] = np.sum(labels == 2)
    class_statics[3] = np.sum(labels == 3)
    class_statics[4] = np.sum(labels == 4)
    print('class_statics is ', class_statics)
    for idx, (nc_roi_path, art_roi_path, pv_roi_path) in enumerate(zip(nc_roi_paths, art_roi_paths, pv_roi_paths)):
        nc_roi_img = cv2.imread(nc_roi_path)
        art_roi_img = cv2.imread(art_roi_path)
        pv_roi_img = cv2.imread(pv_roi_path)
        mean_nc += np.mean(nc_roi_img, (0, 1))
        mean_art += np.mean(art_roi_img, (0, 1))
        mean_pv += np.mean(pv_roi_img, (0, 1))
        if idx / 100 == 0 and idx != 0:
            print(mean_nc / idx, mean_art / idx, mean_pv / idx)

    print('final_mean is ', mean_nc / len(nc_roi_paths), mean_art / len(nc_roi_paths), mean_pv / len(nc_roi_paths))


def read_WC(dicom_dir):
    reader = itk.ImageSeriesReader()
    dicom_series = reader.GetGDCMSeriesFileNames(dicom_dir)
    single_dicom_path = dicom_series[0]
    header = pydicom.read_file(single_dicom_path)
    # print(header.WindowCenter, header.WindowWidth)
    if isinstance(header.WindowCenter, list) and len(header.WindowCenter) != 0:
        return header.WindowCenter[0], header.WindowWidth[0]
    return header.WindowCenter, header.WindowWidth


def resolve_attribute_file(file_path):
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


def get_liver_path(cur_slice_dir, phase_name):
    paths = glob(os.path.join(cur_slice_dir, str(phase_name).lower() + '_liver_mask.mhd'))
    if len(paths) == 1:
        return paths[0]
    paths = glob(os.path.join(cur_slice_dir, str(phase_name).upper() + '_Liver_Mask.mhd'))
    if len(paths) == 1:
        return paths[0]
    print('Error ', cur_slice_dir, phase_name)
    assert False


def get_filled_mask(mask):
    mask = np.asarray(mask == 1, np.uint8)
    mask = fill_region(mask)
    xs, ys = np.where(mask == 1)
    return mask, np.min(xs), np.min(ys), np.max(xs), np.max(ys)


def compute_mean_img(imgs):
    pixel_values = []
    for img in imgs:
        pixel_values.append(img.mean((0, 1)))
    return np.mean(pixel_values, axis=0)


def read_liver_mask(mask_path, size=16):
    from utils.Tools import get_boundingbox
    liver_mask = np.squeeze(read_mhd_image(mask_path))
    min_xs, max_xs, min_ys, max_ys = get_boundingbox(liver_mask)
    center_x = min_xs + (max_xs - min_xs) // 2
    center_y = min_ys + (max_ys - min_ys) // 2
    new_min_xs = center_x - size // 2
    new_max_xs = center_x + size // 2
    new_min_ys = center_y - size // 2
    new_max_ys = center_y + size // 2
    return new_min_xs, new_min_ys, new_max_xs, new_max_ys


def statics_medical_image_attributes(dataset_dir):
    '''
    这是最后的结果， 55.816561844863735, 57.14989517819706, 228.17085953878407, 3554.8784067085953, 0.6874831037735849
    236, 209, 967, 33974, 1.096961
    :param dataset_dir:
    :return:
    '''
    slice_names = os.listdir(dataset_dir)
    heights = []
    widths = []
    perimeters = []
    areas = []
    circle_rates = []
    for slice_name in slice_names:
        cur_dataset_dir = os.path.join(dataset_dir, slice_name)
        nc_attribute_path = os.path.join(cur_dataset_dir, 'NC_attributes.txt')
        art_attribute_path = os.path.join(cur_dataset_dir, 'ART_attributes.txt')
        pv_attribute_path = os.path.join(cur_dataset_dir, 'PV_attributes.txt')
        nc_attris = resolve_attribute_file(nc_attribute_path)
        pv_attris = resolve_attribute_file(pv_attribute_path)
        art_attris = resolve_attribute_file(art_attribute_path)
        heights.extend([nc_attris[0], art_attris[0], pv_attris[0]])
        widths.extend([nc_attris[1], art_attris[1], pv_attris[1]])
        perimeters.extend([nc_attris[2], art_attris[2], pv_attris[2]])
        areas.extend([nc_attris[3], art_attris[3], pv_attris[3]])
        circle_rates.extend([nc_attris[4], art_attris[4], pv_attris[4]])
    print(
    'mean height, width, perimeter, area, circle_rate are ', np.mean(heights), np.mean(widths), np.mean(perimeters),
    np.mean(areas), np.mean(circle_rates))
    print(
        'max height, width, perimeter, area, circle_rate are ', np.max(heights), np.max(widths), np.max(perimeters),
        np.max(areas), np.max(circle_rates))


def statics_medical_image_windows(medical_image_dir, save_path):

    lines = []
    for sub_name in ['CYST', 'FNH', 'HCC', 'HEM', 'METS']:
        sub_dir = os.path.join(medical_image_dir, sub_name)
        names = os.listdir(sub_dir)
        for name in names:
            if name.startswith('.'):
                continue
            print(name)
            NC_DIR = os.path.join(sub_dir, name, 'NC')
            NC_WC = read_WC(NC_DIR)
            ART_DIR = os.path.join(sub_dir, name, "ART")
            ART_WC = read_WC(ART_DIR)
            PV_DIR = os.path.join(sub_dir, name, 'PV')
            PV_WC = read_WC(PV_DIR)
            print(NC_WC, ART_WC, PV_WC)
            base_name = name.split('-')[1]
            line = '%s %d %d %d %d %d %d\n' % (
            base_name, NC_WC[0] - NC_WC[1] // 2, NC_WC[0] + NC_WC[1] // 2, ART_WC[0] - ART_WC[1] // 2,
            ART_WC[0] + ART_WC[1] // 2, PV_WC[0] - PV_WC[1] // 2, PV_WC[0] + PV_WC[1] // 2)
            print(line)
            lines.append(line)
    if save_path is not None:
        with open(save_path, 'w') as writed_file:
            writed_file.writelines(lines)
            writed_file.close()


def mv_uncertain(dataset_dir, save_dir):
    uncertain_txt_path = os.path.join(dataset_dir, 'uncertain.txt')
    import shutil
    with open(uncertain_txt_path, 'r') as uncertain_txt_file:
        lines = uncertain_txt_file.readlines()
        cur_state_name = None
        for line  in lines:
            line = line.replace('\n', '')
            if line in ['train', 'val', 'test']:
                cur_state_name = line
                continue
            cur_dir_path = os.path.join(dataset_dir, cur_state_name, line)

            shutil.copytree(cur_dir_path, os.path.join(save_dir, line))


if __name__ == '__main__':
    # for phasename in ['NC', 'ART', 'PV']:
    #     convert_dicomseries2mhd(
    #         '/home/give/github/Cascaded-FCN-Tensorflow/Cascaded-FCN/tensorflow-unet/z_testdata/304176-2802027/' + phasename,
    #         '/home/give/github/Cascaded-FCN-Tensorflow/Cascaded-FCN/tensorflow-unet/z_testdata/304176-2802027/MHD/' + phasename + '.mhd'
    #     )

    # names = os.listdir('/home/give/Documents/dataset/ISBI2017/media/nas/01_Datasets/CT/LITS/Training_Batch_2')
    # for name in names:
    #     path = os.path.join('/home/give/Documents/dataset/ISBI2017/media/nas/01_Datasets/CT/LITS/Training_Batch_2', name)
    #     image = read_nil(path)
    #     print(np.shape(image))


    # conver2JPG single phase
    # image_dir = '/home/give/Documents/dataset/MICCAI2018/Slices/crossvalidation/0'
    # save_dir = '/home/give/Documents/dataset/MICCAI2018_Detection/SinglePhase'
    # phase_name = 'NC'
    # MICCAI2018_Iterator(image_dir, dicom2jpg_singlephase, save_dir, phase_name)

    # calculate_mean_pixel()
    # statics_medical_image_windows('/media/dl-box/HDCZ-UT/datasets/MedicalImage',
    #                               '/media/dl-box/HDCZ-UT/datasets/MedicalImage/window_center_info.txt')


    # 计算attributions的平均值
    # statics_medical_image_attributes(
    #     '/home/dl-box/ld/Documents/datasets/IEEEonMedicalImage_Splited/JPG/0_attribute/train')


    mv_uncertain('/资料/数据集/MICCAI2018/Slices/crossvalidation/0',
                 '/资料/数据集/MICCAI2018/Slices/crossvalidation/0/uncertain')
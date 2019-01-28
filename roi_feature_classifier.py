# -*- coding=utf-8 -*-
import numpy as np
from PIL import Image
import os
from utils.Tools import calculate_acc_error
class_num = 4


def get_label_from_pixelvalue(pixel_value):
    '''
    根据像素值返回预测的label的值
    :param pixel_value:
    :return:
    '''
    if pixel_value[0] >= 200 and pixel_value[1] >= 200 and pixel_value[2] >= 200:
        print 'Error'
        return 1
    if pixel_value[1] >= 200 and pixel_value[2] >= 200:
        return 3
    if pixel_value[0] >= 200:
        return 2
    if pixel_value[1] >= 200:
        return 0
    if pixel_value[2] >= 200:
        return 1


def generate_feature_by_heatingmap(image):
    '''
    产生一幅热力图对应的特征向量
    :param image:　热力图
    :return:对应的特征向量
    '''
    features = np.zeros([1, class_num], np.float32)
    shape = list(np.shape(image))
    for i in range(shape[0]):
        for j in range(shape[1]):
            pixel_value = image[i, j]
            index = get_label_from_pixelvalue(pixel_value)
            features[0, index] += 1
    features /= np.sum(features)
    return np.array(features).squeeze()


def generate_features_multiheatingmap(dir_path):
    names = os.listdir(dir_path)
    image_paths = [os.path.join(dir_path, name) for name in names]
    features = [generate_feature_by_heatingmap(np.array(Image.open(path))) for path in image_paths]
    return features


def generate_features_labels(data_dir):
    '''
    生成data_dir目录下面所有文件的features以及对应的labels
    :param data_dir:
    :return:
    '''
    train_features = []
    train_labels = []
    val_features = []
    val_labels = []
    test_features = []
    test_labels = []
    for subclass in ['train', 'val', 'test']:
        for type in [0, 1, 2, 3]:
            cur_features = generate_features_multiheatingmap(os.path.join(data_dir, subclass, str(type)))
            if subclass == 'train':
                train_features.extend(cur_features)
                train_labels.extend([type] * len(cur_features))
            elif subclass == 'val':
                val_features.extend(cur_features)
                val_labels.extend([type] * len(cur_features))
            else:
                test_features.extend(cur_features)
                test_labels.extend([type] * len(cur_features))
    scio.savemat('data.mat', {
        'train_features': train_features,
        'train_labels': train_labels,
        'val_features': val_features,
        'val_labels': val_labels,
        'test_features': test_features,
        'test_labels': test_labels
    })
    return train_features, train_labels, val_features, val_labels, test_features, test_labels


def load_feature(dataset_dir='/media/dl-box/HDD3/ld/Documents/datasets/IEEEonMedicalImage_Splited/0/roi_feature/7x7/res50_original_decay_lr',
                 basename='res50'):
    from dataset.medicalImage import resolve_attribute_file
    import config
    def _load_label_with_attributions(dataset_dir, stage_name):
        slice_names = os.listdir(os.path.join(dataset_dir, stage_name))
        labels = []
        nc_attrs = []
        art_attrs = []
        pv_attrs = []
        for slice_name in slice_names:
            labels.append(int(slice_name[-1]))
            nc_attr_path = os.path.join(dataset_dir, stage_name, slice_name, 'NC_attributes.txt')
            art_attr_path = os.path.join(dataset_dir, stage_name, slice_name, 'ART_attributes.txt')
            pv_attr_path = os.path.join(dataset_dir, stage_name, slice_name, 'PV_attributes.txt')
            nc_attr = resolve_attribute_file(nc_attr_path)
            art_attr = resolve_attribute_file(art_attr_path)
            pv_attr = resolve_attribute_file(pv_attr_path)
            nc_attrs.append(nc_attr)
            art_attrs.append(art_attr)
            pv_attrs.append(pv_attr)
        nc_attrs = np.asarray(nc_attrs, np.float32)
        art_attrs = np.asarray(art_attrs, np.float32)
        pv_attrs = np.asarray(pv_attrs, np.float32)
        nc_attrs /= config.MAX_ATTRS
        art_attrs /= config.MAX_ATTRS
        pv_attrs /= config.MAX_ATTRS
        attrs = np.concatenate([nc_attrs, art_attrs, pv_attrs], axis=1)
        return labels, slice_names, attrs

    def _load_label(dataset_dir='/media/dl-box/HDD3/ld/Documents/datasets/IEEEonMedicalImage_Splited/0/temp', stage_name='train'):
        print(os.path.join(dataset_dir, stage_name))
        slice_names = os.listdir(os.path.join(dataset_dir, stage_name))
        labels = []
        res_slice_names = []
        for slice_name in slice_names:
            if slice_name.startswith('.DS'):
                continue
            labels.append(int(slice_name[-1]))
            res_slice_names.append(slice_name)
        print slice_names
        return labels, res_slice_names

    train_npy_path = os.path.join(dataset_dir, basename + '_train.npy')
    val_npy_path = os.path.join(dataset_dir, basename + '_val.npy')
    test_npy_path = os.path.join(dataset_dir, basename + '_test.npy')
    train_labels, train_slice_names = _load_label(stage_name='train')
    val_labels, val_slice_names = _load_label(stage_name='val')
    test_labels, test_slice_names = _load_label(stage_name='test')
    train_features = np.load(train_npy_path)
    val_features = np.load(val_npy_path)
    test_features = np.load(test_npy_path)
    # train_features = np.concatenate([train_features, train_attrs], axis=1)
    # val_features = np.concatenate([val_features, val_attrs], axis=1)
    # test_features = np.concatenate([test_features, test_attrs], axis=1)
    print(np.shape(train_features), np.shape(val_features), np.shape(test_features), len(train_labels), len(val_labels),
          len(test_labels))
    return train_features, train_labels, val_features, val_labels, test_features, test_labels,\
           train_slice_names, val_slice_names, test_slice_names


if __name__ == '__main__':
    train_features, train_labels, val_features, val_labels, test_features, test_labels, \
    train_slice_names, val_slice_names, test_slice_names = load_feature()
    for test_feature, test_slice_name in zip(test_features, test_slice_names):
        print test_feature, test_slice_name
    import scipy.io as scio
    from utils.classification import SVM, KNN

    # train_data = scio.loadmat('./features/crossvalidation/0/saved/train.npy.mat')
    # train_features = train_data['features']
    # train_labels = train_data['labels']
    #
    # val_data = scio.loadmat('./features/crossvalidation/0/saved/val.npy.mat')
    # val_features = val_data['features']
    # val_labels = val_data['labels']
    #
    # test_data = scio.loadmat('./features/crossvalidation/0/saved/test.npy.mat')
    # test_features = test_data['features']
    # test_labels = test_data['labels']
    # SVM
    predicted_label, c_params, g_params, accs = SVM.do(train_features, train_labels, test_features, test_labels,
                                                       adjust_parameters=True)
    # use default parameters
    predicted_label, acc = SVM.do(train_features, train_labels, test_features, test_labels, adjust_parameters=False,
                                  C=1.0, gamma='auto')
                                  # C=c_params, gamma=g_params)
    # predicted_label = np.argmax(test_features, axis=1)
    # print 'ACC is ', acc
    calculate_acc_error(predicted_label, test_labels)
    for idx in range(len(test_labels)):
        if predicted_label[idx] != test_labels[idx]:
            print('prediction error, gt is %d, pred is %d, slice_name is %s' %
                  (test_labels[idx], predicted_label[idx], test_slice_names[idx]))

    # KNN
    _, k = KNN.do(train_features, train_labels, val_features, val_labels,
                  adjust_parameters=True)
    # use default parameters
    predicted_label = KNN.do(train_features, train_labels, test_features, test_labels, adjust_parameters=False,
                             k=k)
    calculate_acc_error(predicted_label, test_labels)
    for idx in range(len(test_labels)):
        if predicted_label[idx] != test_labels[idx]:
            print('prediction error, gt is %d, pred is %d, slice_name is %s' %
                  (test_labels[idx], predicted_label[idx], test_slice_names[idx]))
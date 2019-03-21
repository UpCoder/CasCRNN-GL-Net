import numpy
import os


def statics_case_num(dataset_dir):
    def _core(dataset_dir):
        case_dict = [{}, {}, {}, {}, {}]
        names = os.listdir(dataset_dir)
        for name in names:
            if not os.path.isdir(os.path.join(dataset_dir, name)):
                continue
            case_id = '_'.join(name.split('_')[:2])
            # print(case_id)
            category_id = int(name[-1])
            if case_id in case_dict[category_id].keys():
                continue
            else:
                case_dict[category_id][case_id] = 1
        for idx, category_name in enumerate(['CYST', 'FNH', 'HCC', 'HEM', 'METS']):
            print(category_name, 'case num is ', len(case_dict[idx].keys()))

    for sub_dir in ['train','val','test']:
        print(sub_dir)
        cur_data_dir = os.path.join(dataset_dir, sub_dir)
        _core(cur_data_dir)


if __name__ == '__main__':
    dataset_dir = '/Volumes/ELEMENTS/Documents/datasets/IEEEonMedicalImage_Splited/0'
    statics_case_num(dataset_dir)
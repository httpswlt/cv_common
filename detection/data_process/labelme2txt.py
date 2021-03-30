# coding:utf-8
import json
import os
import random
import re
import shutil

import numpy as np


class Labelme2Txt:
    def __init__(self, classes=None):
        self.classes = classes
        self.rm_txt = False
        self.label_txt_suffix = 'txt'
        self.total_datasets_count = {}

    def set_classes(self, classes):
        """

        :param classes:
        :return:
        """
        self.classes = classes

    def set_rm_txt(self, rm_txt):
        """

        :param rm_txt:
        :return:
        """
        self.rm_txt = rm_txt

    def print_total_datasets_info(self):
        """

        :return:
        """
        img_total_nums = 0
        label_total_nums = 0
        for classify, info in self.total_datasets_count:
            label_nums = info['label_num']
            img_nums = len(info['imgs'])
            print("{} image nums: {}, label nums: {}".format(classify, img_nums, label_nums))
            img_total_nums += img_nums
            label_total_nums += label_nums
        print("Total Image Num: {}, Label Num: {}".format(img_total_nums, label_total_nums))

    @staticmethod
    def convert(size, box):
        """

        :param size:
        :param box:
        :return:
        """
        dw = 1. / (size[0])
        dh = 1. / (size[1])
        x = (box[0] + box[1]) / 2.0 - 1
        y = (box[2] + box[3]) / 2.0 - 1
        w = box[1] - box[0]
        h = box[3] - box[2]
        x = x * dw
        w = w * dw
        y = y * dh
        h = h * dh
        return x, y, w, h

    def __count_imgs_labels_num(self, label, img_path):
        """

        :param label:
        :param img_path:
        :return:
        """
        data_info = self.total_datasets_count.get(label, {})
        tmp = data_info.get('label_num', 0)
        tmp += 1
        data_info['label_num'] = tmp
        self.total_datasets_count[label] = data_info
        tmp = data_info.get('imgs', [])
        tmp.append(img_path)
        data_info['imgs'] = list(set(tmp))

    def json_2_txt(self, json_path):
        """

        :param json_path:
        :return:
        """
        assert self.classes is not None and len(self.classes) != 0
        json_result = json.load(open(json_path, "r", encoding="utf-8"))
        base_dir = os.path.dirname(json_path)
        json_name = os.path.basename(json_path)[:-5]
        label_txt = os.path.join(base_dir, "{}.{}".format(json_name, self.label_txt_suffix))
        if self.rm_txt:
            if os.path.exists(label_txt):
                os.remove(label_txt)
            return
        out_file = open(label_txt, 'w')
        h, w = json_result['imageHeight'], json_result['imageWidth']
        for multi in json_result["shapes"]:
            points = np.array(multi["points"])
            xmin = min(points[:, 0]) if min(points[:, 0]) > 0 else 0
            xmax = max(points[:, 0]) if max(points[:, 0]) > 0 else 0
            ymin = min(points[:, 1]) if min(points[:, 1]) > 0 else 0
            ymax = max(points[:, 1]) if max(points[:, 1]) > 0 else 0
            if xmax <= xmin:
                pass
            elif ymax <= ymin:
                pass
            else:
                label = multi["label"]
                if label not in self.classes:
                    label = re.sub('_\d+', '', label)
                cls_id = self.classes.index(label)
                b = (float(xmin), float(xmax), float(ymin), float(ymax))
                bb = self.convert((w, h), b)
                out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

    def json_2_txt_dir(self, json_dir):
        """

        :param json_dir:
        :return:
        """
        for j_file in os.listdir(json_dir):
            if not j_file.endswith('.json'):
                continue
            json_path = os.path.join(json_dir, j_file)
            self.json_2_txt(json_path)

    def count_classes(self, json_dir, merge=False, assign_print=None, form='dict'):
        """

        :param json_dir:
        :param merge:
        :param assign_print:
        :return:
        """
        classes = []
        self.__items_classes(json_dir, classes, assign_print=assign_print)
        classes = sorted(list(set(classes)))
        if merge:
            classes = [re.sub('_\d+', '', classify) for classify in classes]
        classes = sorted(list(set(classes)))
        if 'dict' == form:
            classes = {classes[i - 1]: i for i in range(1, len(classes) + 1)}
            return classes
        elif 'list' == form:
            return classes

    def __items_classes(self, json_dir, classes, assign_print=None):
        """

        :param json_dir:
        :param classes:
        :param assign_print:
        :return:
        """
        for j_file in os.listdir(json_dir):
            j_path = os.path.join(json_dir, j_file)
            if not os.path.isfile(j_path):
                self.__items_classes(j_path, classes, assign_print)
            if not j_file.endswith('.json'):
                continue
            json_path = os.path.join(json_dir, j_file)
            json_result = json.load(open(json_path, "r", encoding="utf-8"))
            for multi in json_result["shapes"]:
                label = multi["label"]
                if assign_print is not None and assign_print in label:
                    print("label: {}: Path: {}".format(label, json_path))
                classes.append(label)

    def copy_training_data(self, src_dir, dist_dir):
        # count all labels
        targets = []
        for json_file in os.listdir(src_dir):
            if not json_file.endswith(self.label_txt_suffix):
                continue
            targets.append(os.path.join(src_dir, json_file))
        training_num = int(len(targets) * 0.9)
        random.shuffle(targets)
        training_sets = targets[:training_num]
        val_sets = targets[training_num:]
        img_dir = os.path.join(dist_dir, 'images')
        label_dir = os.path.join(dist_dir, 'labels')
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)
        datasets = {
            'train': training_sets,
            'val': val_sets
        }
        for key, value in datasets.items():
            type_img_path = os.path.join(img_dir, key)
            type_label_path = os.path.join(label_dir, key)
            os.makedirs(type_img_path, exist_ok=True)
            os.makedirs(type_label_path, exist_ok=True)
            for label_path in value:
                img_name = os.path.basename(label_path).split('.{}'.format(self.label_txt_suffix))[0]
                img_path = os.path.join(src_dir, '{}.jpg'.format(img_name))
                if not os.path.exists(img_path):
                    img_path = os.path.join(type_img_path, '{}.png'.format(img_name))
                if os.path.exists(label_path) and os.path.exists(img_path):
                    shutil.copy(label_path, type_label_path)
                    shutil.copy(img_path, type_img_path)
                else:
                    print(label_path, img_path)
                    exit(0)


def main():
    convert_var = Labelme2Txt()
    json_dir = '/mnt/data/logo/training'
    # classes = convert_var.count_classes(json_dir, merge=True, assign_print='MJHP')
    classes = convert_var.count_classes(json_dir, merge=False, form='list')
    # origin_classes = ['apollo', 'BBC', 'epoch', 'Formosa TV News network', 'FTchinese', 'NTD']
    # classes = origin_classes + classes

    for i, classify in classes.items():
        print("{}: {}".format(i, classify))
    print(classes)
    print("total classify: {}".format(len(classes)))

    # convert_var.set_classes(classes)
    # # convert_var.set_rm_txt(True)
    # # exit(0)
    # new_path = '/home/lintao/datasets/logo_det/logo_{}'.format(len(classes))
    # if os.path.exists(new_path):
    #     os.system('rm -rf {}'.format(new_path))
    # print("Copy Label Data to {} ".format(new_path))
    # for i, classify in enumerate(os.listdir(json_dir)):
    #     print(classify)
    #     classify_path = os.path.join(json_dir, classify)
    #     # convert_var.json_2_txt_dir(classify_path)
    #     convert_var.copy_training_data(classify_path, new_path)


if __name__ == '__main__':
    main()

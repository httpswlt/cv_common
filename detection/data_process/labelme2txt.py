# coding:utf-8
import json
import os
import random
import re
import shutil
import glob
import numpy as np


class Labelme2Txt:
    def __init__(self, json_list_path):
        """

        :param json_list_path:
        :param image_dir:
        """
        self.json_list_path = json_list_path

        self.total_datasets_count = {}
        self.label_txt_suffix = 'txt'
        self.classes = None
        self.dtype = 'train'

    def set_classes(self, classes):
        """

        :param classes:
        :return:
        """
        self.classes = classes

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

    def count_classes(self, json_dir, merge=False, assign_print=None, form='list'):
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

    def set_data_type(self, dtype):
        self.dtype = 'train'

    def to_txt(self, save_dir, copy_img=False):
        """

        :param save_dir:
        :param copy_img:
        :return:
        """
        print("Copy Label Data to {} ".format(save_dir))
        label_save_dir = os.path.join(save_dir, 'labels', self.dtype)
        image_save_dir = os.path.join(save_dir, 'images', self.dtype)
        os.makedirs(label_save_dir, exist_ok=True)
        os.makedirs(image_save_dir, exist_ok=True)
        for json_path in self.json_list_path:
            assert self.classes is not None and len(self.classes) != 0
            json_result = json.load(open(json_path, "r", encoding="utf-8"))
            json_suffix = os.path.basename(json_path).split('.')
            json_suffix[-1] = self.label_txt_suffix
            json_name = '.'.join(json_suffix)
            label_txt = os.path.join(label_save_dir, json_name)

            with open(label_txt, 'w') as f:
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
                        f.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
            if copy_img:
                json_suffix[-1] = 'jpg'
                img_name = '.'.join(json_suffix)
                img_path = os.path.join(os.path.dirname(json_path), img_name)
                if not os.path.exists(img_path):
                    json_suffix[-1] = 'png'
                    img_name = '.'.join(json_suffix)
                    img_path = os.path.join(os.path.dirname(json_path), img_name)
                shutil.copy(img_path, image_save_dir)
            break


def main():
    # json_dir = '/home/lintao/docker_share/logo_data/labelme'
    json_dir = '/mnt/data/logo/training'
    save_path = '/home/lintao/docker_share/logo_data/txt'
    json_path = glob.glob(os.path.join(json_dir, "*/*.json"))

    convert_var = Labelme2Txt(json_path)
    classes = convert_var.count_classes(json_dir, merge=False)
    convert_var.set_classes(classes)
    convert_var.to_txt(save_path)


if __name__ == '__main__':
    main()

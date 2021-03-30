# coding:utf-8
import os
import json
import numpy as np
import glob
import shutil
from labelme import utils

from sklearn.model_selection import train_test_split

np.random.seed(41)

classname_to_id = {'BowenPress_1': 1, 'BowenPress_2': 2, 'FTchinese': 3, 'Formosa TV News network': 4, 'RFA': 5,
                   'apollo': 6, 'bbc_1': 7, 'cnn_1': 8, 'creaders_0': 9, 'creaders_1': 10, 'creaders_2': 11,
                   'cti_0': 12, 'cti_2': 13, 'epoch_1': 14, 'epoch_2': 15, 'global_videw_1': 16, 'global_videw_2': 17,
                   'jgzs_1': 18, 'jgzs_2': 19, 'mjhp_0': 20, 'mjhp_1': 21, 'mjhp_2': 22, 'ntd_0': 23, 'ntd_1': 24,
                   'ntd_2': 25, 'voa_0': 26, 'zyzg_0': 27, 'zyzg_2': 28}


class Lableme2CoCo:

    def __init__(self):
        self.images = []
        self.annotations = []
        self.categories = []
        self.img_id = 0
        self.ann_id = 0

    def save_coco_json(self, instance, save_path):
        json.dump(instance, open(save_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)  # indent=2 更加美观显示

    def to_coco(self, json_path_list):
        self._init_categories()
        for json_path in json_path_list:
            obj = self.read_jsonfile(json_path)
            # self.images.append(self._image(obj, json_path))
            self.images.append(self._image_no_data(obj, json_path))
            shapes = obj['shapes']
            for shape in shapes:
                annotation = self._annotation(shape)
                self.annotations.append(annotation)
                self.ann_id += 1
            self.img_id += 1
        instance = {}
        instance['info'] = 'spytensor created'
        instance['license'] = ['license']
        instance['images'] = self.images
        instance['annotations'] = self.annotations
        instance['categories'] = self.categories
        return instance

    def _init_categories(self):
        for k, v in classname_to_id.items():
            category = {}
            category['id'] = v
            category['name'] = k
            self.categories.append(category)

    def _image(self, obj, path):
        image = {}
        img_x = utils.img_b64_to_arr(obj['imageData'])
        h, w = img_x.shape[:-1]
        image['height'] = h
        image['width'] = w
        image['id'] = self.img_id
        image['file_name'] = os.path.basename(path).replace(".json", ".jpg")
        return image

    def _image_no_data(self, obj, path):
        image = {}
        image['height'] = obj['imageHeight']
        image['width'] = obj['imageWidth']
        image['id'] = self.img_id
        image['file_name'] = os.path.basename(path).replace(".json", ".jpg")
        return image

    def _annotation(self, shape):
        label = shape['label']
        points = shape['points']
        annotation = {}
        annotation['id'] = self.ann_id
        annotation['image_id'] = self.img_id
        annotation['category_id'] = int(classname_to_id[label])
        annotation['segmentation'] = self._get_segmentation(points)
        annotation['bbox'] = self._get_box(points)
        annotation['iscrowd'] = 0
        annotation['area'] = 1.0
        return annotation

    def read_jsonfile(self, path):
        with open(path, "r", encoding='utf-8') as f:
            return json.load(f)

    def _get_box(self, points):
        min_x = min_y = np.inf
        max_x = max_y = 0
        for x, y in points:
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
        return [min_x, min_y, max_x - min_x, max_y - min_y]

    def _get_segmentation(self, points):
        x1 = points[0][0]
        y1 = points[0][1]
        x2 = points[1][0]
        y2 = points[1][1]
        seg = [list(np.asarray([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]).flatten())]
        # seg = [np.asarray(points).flatten().tolist()]
        return seg


if __name__ == '__main__':
    labelme_path = "/mnt/data/logo/training/"
    saved_coco_path = "/home/lintao/docker_share/logo_coco/logo/"
    if not os.path.exists("%sannotations/" % saved_coco_path):
        os.makedirs("%sannotations/" % saved_coco_path)
    if not os.path.exists("%simages/train2017/" % saved_coco_path):
        os.makedirs("%simages/train2017" % saved_coco_path)
    if not os.path.exists("%simages/val2017/" % saved_coco_path):
        os.makedirs("%simages/val2017" % saved_coco_path)

    json_list_path = glob.glob(labelme_path + "*/*.json")

    train_path, val_path = train_test_split(json_list_path, test_size=0.12)
    print("train_n:", len(train_path), 'val_n:', len(val_path))

    l2c_train = Lableme2CoCo()
    train_instance = l2c_train.to_coco(train_path)
    l2c_train.save_coco_json(train_instance, '%sannotations/instances_train2017.json' % saved_coco_path)
    # for file in train_path:
    #     shutil.copy(file.replace("json", "jpg"), "%simages/train2017/" % saved_coco_path)
    # for file in val_path:
    #     shutil.copy(file.replace("json", "jpg"), "%simages/val2017/" % saved_coco_path)

    l2c_val = Lableme2CoCo()
    val_instance = l2c_val.to_coco(val_path)
    l2c_val.save_coco_json(val_instance, '%sannotations/instances_val2017.json' % saved_coco_path)

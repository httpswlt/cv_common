# -*- coding: utf-8 -*- #
# --------------------------------------------------------
# Author: Lintao
# Created: 2020/03/31
# --------------------------------------------------------
from pascal_voc_writer import Writer
from pycocotools.coco import COCO
import os
from tqdm import tqdm
import shutil
import json


class COCO2Labelme:
    def __init__(self, json_path, images_dir):
        self.coco_var = COCO(json_path)
        self.images_dir = images_dir
        self.shape = None
        self.labelme_dict = None
        self.init_labelme_dict()

    def init_labelme_dict(self):
        self.shape = {
            "label": "test_0",
            "points": [['x1', 'y1'], ['x2', 'y2']],
            "group_id": None,
            "shape_type": "rectangle",
            "flags": {}
        }
        self.labelme_dict = {
            "version": "4.5.6",
            "flags": {},
            "shapes": [self.shape],
            "imagePath": "test.jpg",
            "imageData": None,
            "imageHeight": 720,
            "imageWidth": 1280
        }

    def to_labelme(self, save_dir, copy_img=False):
        """

        :param save_dir:
        :param copy_img: whether copy the image from ori-path to new path.
        :return:
        """
        os.makedirs(save_dir, exist_ok=True)
        cats = self.coco_var.loadCats(self.coco_var.getCatIds())
        cat_idx = {}
        for c in cats:
            cat_idx[c['id']] = c['name']
        for img in tqdm(self.coco_var.imgs):
            cat_ids = self.coco_var.getCatIds()
            ann_ids = self.coco_var.getAnnIds(imgIds=[img], catIds=cat_ids)
            if len(ann_ids) > 0:
                image_info = self.coco_var.imgs[img]
                image_suffix = image_info['file_name'].split('.')
                image_suffix[-1] = 'json'
                file_name = '.'.join(image_suffix)
                self.labelme_dict['imagePath'] = image_info['file_name']
                self.labelme_dict['imageHeight'] = image_info['height']
                self.labelme_dict['imageWidth'] = image_info['width']
                shapes = []
                anns = self.coco_var.loadAnns(ann_ids)
                for a in anns:
                    shape = self.shape.copy()
                    bbox = a['bbox']
                    shape['points'] = [[bbox[0], bbox[1]],
                                       [bbox[2] + bbox[0], bbox[3] + bbox[1]]]
                    shape['label'] = cat_idx[a['category_id']]
                    shapes.append(shape)
                self.labelme_dict['shapes'] = shapes

                with open(os.path.join(save_dir, file_name), 'w') as f:
                    json.dump(self.labelme_dict, f)

                if copy_img:
                    image_path = os.path.join(self.images_dir, image_info['file_name'])
                    shutil.copy(image_path, save_dir)


def test1():
    json_path = "/home/lintao/docker_share/logo_data/coco/annotations/instances_val2017.json"
    images_dir = "/home/lintao/docker_share/logo_data/coco/images/val2017"
    json_save_path = '/home/lintao/docker_share/logo_data/labelme/val'
    coco_var = COCO2Labelme(json_path, images_dir)
    coco_var.to_labelme(json_save_path)


if __name__ == '__main__':
    test1()

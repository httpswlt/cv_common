# -*- coding: utf-8 -*- #
# --------------------------------------------------------
# Author: Lintao
# Created: 2020/03/30
# --------------------------------------------------------
from pascal_voc_writer import Writer
from pycocotools.coco import COCO
import os
from tqdm import tqdm
import shutil


class COCO2VOC:
    def __init__(self, json_path, images_dir):
        self.coco_var = COCO(json_path)
        self.images_dir = images_dir

    def to_voc(self, save_dir, copy_img=False):
        """

        :param save_dir:
        :param copy_img: whether copy the image from ori-path to new path.
        :return:
        """
        xml_save_dir = os.path.join(save_dir, 'xml')
        img_save_dir = os.path.join(save_dir, 'images')
        os.makedirs(xml_save_dir, exist_ok=True)
        os.makedirs(img_save_dir, exist_ok=True)
        cats = self.coco_var.loadCats(self.coco_var.getCatIds())
        cat_idx = {}
        for c in cats:
            cat_idx[c['id']] = c['name']
        for img in tqdm(self.coco_var.imgs):
            cat_ids = self.coco_var.getCatIds()
            ann_ids = self.coco_var.getAnnIds(imgIds=[img], catIds=cat_ids)
            if len(ann_ids) > 0:
                image_name = self.coco_var.imgs[img]['file_name']
                image_suffix = image_name.split('.')
                image_suffix[-1] = 'xml'
                label_name = '.'.join(image_suffix)
                image_path = os.path.join(self.images_dir, image_name)
                writer = Writer(image_path, self.coco_var.imgs[img]['width'], self.coco_var.imgs[img]['height'])
                anns = self.coco_var.loadAnns(ann_ids)
                for a in anns:
                    bbox = a['bbox']
                    bbox = [bbox[0], bbox[1], bbox[2] + bbox[0], bbox[3] + bbox[1]]
                    bbox = [str(b) for b in bbox]
                    cat_name = cat_idx[a['category_id']]
                    writer.addObject(cat_name, bbox[0], bbox[1], bbox[2], bbox[3])
                writer.save(os.path.join(xml_save_dir, label_name))
                if copy_img:
                    shutil.copy(image_path, img_save_dir)


def test1():
    json_path = "/home/lintao/docker_share/logo_data/coco/annotations/instances_val2017.json"
    images_dir = "/home/lintao/docker_share/logo_data/coco/images/val2017"
    xml_save_path = '/home/lintao/docker_share/logo_data/voc/val'
    coco_var = COCO2VOC(json_path, images_dir)
    coco_var.to_voc(xml_save_path)


if __name__ == '__main__':
    test1()

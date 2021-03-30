# -*- coding: utf-8 -*- #
# --------------------------------------------------------
# Author: Lintao
# Created: 2020/03/30
# --------------------------------------------------------
import os


def acquire_images(img_dir, imgs):
    if os.path.isfile(img_dir):
        imgs.append(img_dir)
        return

    for img in os.listdir(img_dir):
        acquire_images(os.path.join(img_dir, img), imgs)

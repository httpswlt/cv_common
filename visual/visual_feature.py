# coding:utf-8
from torchvision.utils import make_grid
import numpy as np
from PIL import Image
import os


def visual_layer(tensor, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # if tensor.
    tensor = tensor.permute((1, 0, 2, 3)).cpu()
    im = make_grid(tensor, normalize=True, scale_each=True, nrow=8, padding=2).permute((1, 2, 0))
    im = (im.data.numpy() * 255.).astype(np.uint8)
    Image.fromarray(im).save(save_path)

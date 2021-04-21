# -*- coding: utf-8 -*- #
# --------------------------------------------------------
# Author: Lintao
# Created: 2020/03/30
# --------------------------------------------------------
import os

import cv2
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import torch

if trt.__version__ >= '5.1':
    DEFAULT_CALIBRATION_ALGORITHM = trt.CalibrationAlgoType.ENTROPY_CALIBRATION_2
else:
    DEFAULT_CALIBRATION_ALGORITHM = trt.CalibrationAlgoType.ENTROPY_CALIBRATION


class TensorBatchDataset:

    def __init__(self, tensors):
        self.tensors = tensors

    def __len__(self):
        return len(self.tensors[0])

    def __getitem__(self, idx):
        return [t[idx] for t in self.tensors]


class Calibrator(trt.IInt8Calibrator):
    def __init__(self):
        super(Calibrator, self).__init__()
        self.algorithm = None
        self.batch_size = None
        self.cache_file = './calibrator.cache'

    def get_algorithm(self):
        return self.algorithm

    def get_batch_size(self):
        return self.batch_size

    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)


class DatasetCalibrator(Calibrator):

    def __init__(self, inputs, dataset, batch_size=1, algorithm=DEFAULT_CALIBRATION_ALGORITHM):
        super(DatasetCalibrator, self).__init__()

        self.dataset = dataset
        self.batch_size = batch_size
        self.algorithm = algorithm

        # create buffers that will hold data batches
        self.buffers = []
        for tensor in inputs:
            size = (batch_size,) + tuple(tensor.shape[1:])
            buf = torch.zeros(size=size, dtype=tensor.dtype, device=tensor.device).contiguous()
            self.buffers.append(buf)

        self.count = 0

    def get_batch(self, *args, **kwargs):
        if self.count < len(self.dataset):

            for i in range(self.batch_size):

                idx = self.count % len(self.dataset)  # roll around if not multiple of dataset
                inputs = self.dataset[idx]

                # copy data for (input_idx, dataset_idx) into buffer
                for buffer, tensor in zip(self.buffers, inputs):
                    buffer[i].copy_(tensor)

                self.count += 1

            return [int(buf.data_ptr()) for buf in self.buffers]
        else:
            return []


class ImageCalibrator(Calibrator):

    def __init__(self, images, width=256, height=256, channel=3, batch_size=1, cache_file='./{}.cache'.format('int8')):
        """

        :param images: type: list, e.g: [img1.jpg, img2.jpg, ...]
        :param width:
        :param height:
        :param channel:
        :param batch_size:
        :param cache_file:
        """
        super(ImageCalibrator, self).__init__()
        self.cache_file = cache_file
        self.batch_size = batch_size
        self.channel = channel
        self.height = height
        self.width = width

        assert isinstance(images, list) and len(images) > 0
        self.imgs = images
        self.batch_idx = 0
        self.max_batch_idx = len(self.imgs) // self.batch_size
        self.data_size = trt.volume([self.batch_size, self.channel, self.height, self.width]) * trt.float32.itemsize
        self.device_input = cuda.mem_alloc(self.data_size)
        self.one_batch = self.batch_generator()

    def img_process(self, img_path):
        """

        :param img_path: the path of image
        :return:
        """
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.height, self.width), cv2.INTER_NEAREST)
        img = img / 255.
        img = img.astype(np.float32)
        return img

    def batch_generator(self):
        for i in range(self.max_batch_idx):
            try:
                batch_files = self.imgs[self.batch_idx * self.batch_size: \
                                        (self.batch_idx + 1) * self.batch_size]
                batch_imgs = np.zeros((self.batch_size, self.height, self.width, self.channel),
                                      dtype=np.float32)

                for i, f in enumerate(batch_files):
                    img = self.img_process(f)
                    assert (img.nbytes == self.data_size / self.batch_size), 'not valid img!' + f
                    batch_imgs[i] = img

                self.batch_idx += 1
                print("batch:[{}/{}]".format(self.batch_idx, self.max_batch_idx))
                yield np.ascontiguousarray(batch_imgs)
            except Exception as e:
                print(e)

    def get_batch(self, *args, **kwargs):
        try:
            batch_imgs = next(self.one_batch)
            if batch_imgs.size == 0 or batch_imgs.size != self.batch_size * self.channel * self.height * self.width:
                return None
            cuda.memcpy_htod(self.device_input, batch_imgs.astype(np.float32))
            return [int(self.device_input)]
        except:
            return None

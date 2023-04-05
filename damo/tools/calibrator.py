import os
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import glob

import ctypes
import logging
logger = logging.getLogger(__name__)
ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_char_p
ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=False, stride=32, return_int=False):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    if not return_int:
        return im, r, (dw, dh)
    else:
        return im, r, (left, top)




"""
There are 4 types calibrator in TensorRT.
trt.IInt8LegacyCalibrator
trt.IInt8EntropyCalibrator
trt.IInt8EntropyCalibrator2
trt.IInt8MinMaxCalibrator
"""

class Calibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, stream, cache_file=""):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.stream = stream
        self.d_input = cuda.mem_alloc(self.stream.calibration_data.nbytes)
        self.cache_file = cache_file
        stream.reset()

    def get_batch_size(self):
        return self.stream.batch_size

    def get_batch(self, names):
        print("######################")
        print(names)
        print("######################")
        batch = self.stream.next_batch()
        if not batch.size:
            return None

        cuda.memcpy_htod(self.d_input, batch)
        return [int(self.d_input)]

    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                logger.info("Using calibration cache to save time: {:}".format(self.cache_file))
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            logger.info("Caching calibration data for future use: {:}".format(self.cache_file))
            f.write(cache)


def precess_image(img_src, img_size, stride):
    '''Process image before image inference.'''
    image = letterbox(img_src, img_size, auto=False, return_int=True)[0]
    # Convert
    image = image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    image = np.ascontiguousarray(image).astype(np.float32)
    return image

class DataLoader:
    def __init__(self, batch_size, batch_num, calib_img_dir, input_w, input_h):
        self.index = 0
        self.length = batch_num
        self.batch_size = batch_size
        self.input_h = input_h
        self.input_w = input_w
        # self.img_list = [i.strip() for i in open('calib.txt').readlines()]
        self.img_list = glob.glob(os.path.join(calib_img_dir, "*.jpg"))
        assert len(self.img_list) > self.batch_size * self.length, \
            '{} must contains more than '.format(calib_img_dir) + str(self.batch_size * self.length) + ' images to calib'
        print('found all {} images to calib.'.format(len(self.img_list)))
        self.calibration_data = np.zeros((self.batch_size, 3, input_h, input_w), dtype=np.float32)

    def reset(self):
        self.index = 0

    def next_batch(self):
        if self.index < self.length:
            for i in range(self.batch_size):
                assert os.path.exists(self.img_list[i + self.index * self.batch_size]), 'not found!!'
                img = cv2.imread(self.img_list[i + self.index * self.batch_size])
                img = precess_image(img, self.input_h, 32)
                self.calibration_data[i] = img

            self.index += 1
            return np.ascontiguousarray(self.calibration_data, dtype=np.float32)
        else:
            return np.array([])

    def __len__(self):
        return self.length

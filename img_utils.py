#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File img_utils.py created on 22:39 2017/9/11 

@author: Yichi Xiao
@version: 1.0
"""

import numpy as np
import cv2


def img_crop_to_array(image):
    sprite_size = 32
    img = cv2.resize(image, (sprite_size * 8, sprite_size * 8))
    np_image_data = np.asarray(img, dtype=np.float16)
    np_image_data = cv2.normalize(np_image_data.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    np_data_4d = np.array([]).reshape((0, 32, 32, 3))
    for x in range(8):
        for y in range(8):
            sprite = np_image_data[(x * sprite_size):((x + 1) * sprite_size),
                     (y * sprite_size):((y + 1) * sprite_size)]
            np_data_4d = np.concatenate((np_data_4d, sprite[np.newaxis, ...]), axis=0)
    return np.asarray(np_data_4d, dtype=np.float16)

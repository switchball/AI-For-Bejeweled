#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File img_utils.py created on 22:39 2017/9/11 

@author: Yichi Xiao
@version: 1.0
"""

import numpy as np
import cv2
import os


def img_crop_to_array(image):
    ''' Divide an image into 8x8 small sprites,
        and then convert them into numpy arrays.
        Shape of returned array: (64, 32, 32, 3)
    '''
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

def collect_sprite_training_data(path = "./img_data/"):
    if not path.endswith("/"):
        path = path + "/"
    files = os.listdir(path)
    files = [file for file in files
                if (not os.path.isdir(file))
                and (file.endswith("-data.jpg") or file.endswith("-label.npy"))]
    prefixes = set(file.replace("-data.jpg", "").replace("-label.npy", "") for file in files)
    ret_images = np.array([]).reshape((0, 32, 32, 3))
    ret_labels = np.array([]).reshape((0, ))

    for prefix in prefixes:
        try:
            img_file = cv2.imread(path + prefix + "-data.jpg")
            np_image_data = np.asarray(img_file)
            np_image_data = cv2.normalize(np_image_data.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
            size = int(np_image_data.shape[0] / 32) # normally, it should be 64
            img_array = np_image_data.reshape((size, 32, 32, 3))
            img_label = np.load(path + prefix + '-label.npy')
            # concatenate
            ret_images = np.concatenate((ret_images, img_array), axis=0)
            ret_labels = np.hstack((ret_labels, img_label))
        except Exception as e:
            print(e)
            print("Loading error with prefix =", prefix, ". Skipped.")
            continue

    return ret_images, ret_labels

if __name__ == '__main__':
    features, labels = collect_sprite_training_data()
    import tensorflow as tf
    a, b = tf.train.shuffle_batch([features, labels], batch_size=64,
                                  capacity=64000, min_after_dequeue=10000, enqueue_many=True)
    print(a.shape, b.shape)
    with tf.Session() as sess:
        print(sess.run(b))
        print(sess.run(b))

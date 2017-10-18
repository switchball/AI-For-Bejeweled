#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File SceneLoader.py created on 11:30 2017/9/9 

@author: Yichi Xiao
@version: 1.0
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt


def img2tensor(image):
    np_image_data = np.asarray(image)
    np_image_data = cv2.normalize(np_image_data.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    print(np_image_data.shape)
    print(np_image_data[:, :, 1])


def img_generator(img):
    img = img[45:661, 313:929]
    print(img.shape)
    sprite_size = (int(img.shape[0]/8), int(img.shape[1]/8))
    print(sprite_size)
    for x in range(8):
        for y in range(8):
            sprite = img[(x*sprite_size[0]):((x+1)*sprite_size[0]),
                         (y*sprite_size[1]):((y+1)*sprite_size[1])]
            sprite = cv2.resize(sprite, (32, 32))
            yield sprite



fig = plt.figure()
img = cv2.imread('video/Gem.jpg')
img_array = np.array([]).reshape((0, 32, 32, 3))

for idx, sprite in enumerate(img_generator(img)):
    k = fig.add_subplot(8, 8, idx+1)
    k.imshow(cv2.cvtColor(sprite, cv2.COLOR_BGR2RGB))
    img_array = np.concatenate((img_array, sprite[np.newaxis, ...]), axis=0)
    # img_label = np.append(img_label, [0])

img_label = np.zeros((img_array.shape[0],), dtype=np.int8)
print(img_array.shape)
print(img_label.shape)
np.save('img_data/sample64.npy', img_array)

img_array = np.load('img_data/sample64.npy')
img_label = np.load('img_data/label64.npy') # to do if not exists

plt.xticks([]), plt.yticks([])
#plt.show()


def mouse_event(event, x, y, flags, param):
    global ctrl_mode
    if flags & cv2.EVENT_FLAG_RBUTTON != 0:
        ctrl_mode = 1
    else:
        ctrl_mode = 0
    if event == cv2.EVENT_LBUTTONDOWN:
        idx = int(x/40) + int(y/40) * 8
        if idx >= 64:
            return
        img_label[idx] = crt_label
        print('x=',x,'y=',y,'idx=',idx)



tag_image = np.zeros((320 + 100, 320, 3), np.uint8)
crt_label = 0
ctrl_mode = 0

font = cv2.FONT_HERSHEY_SIMPLEX

cv2.namedWindow('Tagging')
cv2.setMouseCallback('Tagging', mouse_event)
while(1):
    tag_image = np.zeros((320 + 100, 320, 3), np.uint8)
    for idx in range(8 * 8):
        sprite = img_array[idx]
        _x, _y = int(idx / 8), int(idx % 8)
        if ctrl_mode == 0 or img_label[idx] == crt_label:
            tag_image[(_x * 40 + 4):(_x * 40 + 36), (_y * 40 + 4):(_y * 40 + 36)] = sprite
        if img_label[idx] == crt_label and ctrl_mode == 0:
            cv2.rectangle(tag_image, (_y * 40 + 4, _x * 40 + 4), (_y * 40 + 36, _x * 40 + 36), (0, 255, 0), 2)

    cv2.putText(tag_image, 'Current label = %s' % crt_label, (5, 360), font, 1.0, (0, 255, 0), 2)
    cv2.imshow('Tagging', tag_image)
    key = cv2.waitKey(200) & 0xFF
    if (key >= ord('0') and key <= ord('9')):
        crt_label = key - ord('0')
        print('key=', key, 'chr(key)=', chr(key))
    if key == 27:
        break

cv2.destroyAllWindows()
np.save('img_data/label64.npy', img_label)

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File AdaptiveRec.py created on 23:28 2017/10/20 

@author: Yichi Xiao
@version: 1.0
"""

import numpy as np
import cv2
from collections import deque
import time
import os

from Tagging import Tagging

class AdaptiveRec():
    LIMIT = 4
    def __init__(self):
        self.pred_queue = deque()
        self.img_queue = deque()
        self.cnt = np.zeros(64)
        self.picked = deque()

    def append(self, img, preds):
        self.pred_queue.append(preds)
        self.img_queue.append(img)
        if len(self.pred_queue) > AdaptiveRec.LIMIT:
            self.pred_queue.popleft()
        if len(self.img_queue) > AdaptiveRec.LIMIT:
            self.img_queue.popleft()

        self.check()

    def check(self):
        p = self.pred_queue[-1]
        for i in range(64):
            if p[i] == 0:
                self.cnt[i] += 1
            else:
                self.cnt[i] = 0
            if self.cnt[i] == AdaptiveRec.LIMIT:
                for img in self.img_queue:
                    self.pick(img, i)
            if self.cnt[i] > AdaptiveRec.LIMIT:
                self.pick(self.img_queue[-1], i)
        if len(self.picked) >= 64:
            self.save()

    def pick(self, img, index):
        sprite_size = 32
        image = cv2.resize(img, (sprite_size * 8, sprite_size * 8))
        r = int(index / 8)
        c = index % 8
        sprite = image[(r * sprite_size):((r + 1) * sprite_size),
                 (c * sprite_size):((c + 1) * sprite_size)]
        self.picked.append(sprite)

    def show(self):
        tag_image = np.zeros((320 + 100, 320, 3), np.uint8)
        for idx in range(8 * 8):
            if idx >= len(self.picked):
                break
            sprite = cv2.resize(self.picked[idx], (32, 32), cv2.INTER_CUBIC)
            _x, _y = int(idx / 8), int(idx % 8)
            tag_image[(_x * 40 + 4):(_x * 40 + 36), (_y * 40 + 4):(_y * 40 + 36)] = sprite

        cv2.putText(tag_image, 'Current size = %s' % len(self.picked), (10, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.imshow('AdaptiveTag', tag_image)
        cv2.moveWindow('AdaptiveTag', 1024+168, 350)
        cv2.waitKey(10)

    def save(self):
        img_array = np.zeros((32 * 8 * 8, 32, 3), np.uint8)
        for idx in range(64):
            img_array[(idx*32):(idx*32+32), :, :] = self.picked.popleft()

        t = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        cv2.imwrite('img_data/'+t+'-data.jpg', img_array)
        np.save('img_data/'+t+'-label.npy', np.zeros(64))
        print('Done! Saved to img_data/'+t+'-...')
        return t

def collect(path = "./img_data/"):
    if not path.endswith("/"):
        path = path + "/"
    files = os.listdir(path)
    files = [file for file in files
                if (not os.path.isdir(file))
                and (file.endswith("-data.jpg") or file.endswith("-label.npy"))]
    prefixes = set(file.replace("-data.jpg", "").replace("-label.npy", "") for file in files)

    for idx, prefix in enumerate(prefixes):
        try:
            img_file = cv2.imread(path + prefix + "-data.jpg")
            img_label = np.load(path + prefix + '-label.npy')
            np_image_data = np.asarray(img_file)

            if np.sum(img_label > 8) > 0:
                continue

            print('processing {} ({}/{})'.format(prefix, idx, len(prefixes)) )
            tag = Tagging(np_image_data)
            tag.tag(np_image_data, prefix)

            img_label = np.load(path + prefix + '-label.npy')
            print(img_label)

        except FileExistsError as e:
            print(e)
            print("Loading error with prefix =", prefix, ". Skipped.")
            continue


if __name__ == '__main__':
    print("Adaptive Tagging ...")
    collect()
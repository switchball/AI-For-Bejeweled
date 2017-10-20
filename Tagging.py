#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File Tagging.py created on 21:23 2017/9/11 

@author: Yichi Xiao
@version: 1.0
"""

import numpy as np
import cv2
import time

def default_classifier(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    white = cv2.mean(cv2.inRange(hsv, np.array([0, 0, 100]   ), np.array([179, 64, 255])))[0]
    red =   cv2.mean(cv2.inRange(hsv, np.array([170,100, 100]), np.array([179,255, 255])))[0]
    red +=  cv2.mean(cv2.inRange(hsv, np.array([0  ,100, 100]), np.array([0  ,255, 255])))[0]
    orange= cv2.mean(cv2.inRange(hsv, np.array([2,  100, 100]), np.array([20, 255, 255])))[0]
    yellow= cv2.mean(cv2.inRange(hsv, np.array([21, 100, 100]), np.array([32, 255, 255])))[0]
    green = cv2.mean(cv2.inRange(hsv, np.array([60, 100, 100]), np.array([80, 255, 255])))[0]
    blue  = cv2.mean(cv2.inRange(hsv, np.array([80, 100, 100]), np.array([110,255, 255])))[0]
    purple= cv2.mean(cv2.inRange(hsv, np.array([145,100, 100]), np.array([155,255, 255])))[0]
    values = [50, red, orange, yellow+8, green, blue, purple+16, white]
    return np.argmax(np.array(values))


class Tagging:

    def __init__(self, image, default_classifier=default_classifier):
        self.img = np.array(image, copy=True)
        self.img_array = np.array([]).reshape((0, 32, 32, 3))
        self.img_array = np.zeros((32*8*8,32,3), np.uint8)
        self.img_label = np.zeros((64,), dtype=np.int8)
        self.crt_label = 0
        self.ctrl_mode = 0
        self.default_predictor = default_classifier
        self.start()


    @staticmethod
    def img_generator(image):
        sprite_size = 32
        img = cv2.resize(image, (sprite_size*8, sprite_size*8))
        for x in range(8):
            for y in range(8):
                sprite = img[(x*sprite_size):((x+1)*sprite_size),
                             (y*sprite_size):((y+1)*sprite_size)]
                yield sprite

    @staticmethod
    def attach(image, preds=None, fancy=True):
        if preds is None:
            preds = [default_classifier(x) for x in Tagging.img_generator(image)]
        step_h, step_w = int(image.shape[0]/8), int(image.shape[1]/8)
        for idx, pred in enumerate(preds):
            y, x = int(idx/8), idx % 8
            if fancy:
                center = (int((x+.5)*step_w), int((y+.5)*step_h))
                radius = int(0.309 * min(step_h, step_w))
                color = (0, 0, 0) # default black
                color = (255, 0, 0) if pred == 1 else color
                color = (255, 128, 0) if pred == 2 else color
                color = (255, 255, 0) if pred == 3 else color
                color = (0, 255, 0) if pred == 4 else color
                color = (0, 0, 255) if pred == 5 else color
                color = (255, 0, 255) if pred == 6 else color
                color = (255, 255, 255) if pred == 7 else color
                color = (56, 200, 184) if pred == 8 else color
                color = (color[2], color[1], color[0])
                cv2.circle(image, center, radius, color, thickness=2)
            else:
                pass
            cv2.putText(image, '%s' % pred, (22+x * step_w, 42 + y * step_h), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        return image

    def start(self):
        self.img_array = np.zeros((32*8*8,32,3), np.uint8)
        for idx, sprite in enumerate(self.img_generator(self.img)):
            self.img_array[(idx*32):((idx+1)*32), :, :] = sprite
        self.img_label = np.zeros((int(self.img_array.shape[0]/32),), dtype=np.int8)

    def start0(self):
        # for idx, sprite in enumerate(Tagging.img_generator(self.img)):
        #     self.img_array = np.concatenate((self.img_array, sprite[np.newaxis, ...]), axis=0)
        if False:
            new_img = np.reshape(self.img_array, (32*8*8, 32, 3))

    @staticmethod
    def mouse_event(event, x, y, flags, param):
        s = param['self']
        if flags & cv2.EVENT_FLAG_RBUTTON != 0:
            s.ctrl_mode = 1
        else:
            s.ctrl_mode = 0
        if event == cv2.EVENT_LBUTTONDOWN:
            idx = int(x / 80) + int(y / 80) * 8
            if idx >= 64:
                return
            s.img_label[idx] = s.crt_label
            # print('x=', x, 'y=', y, 'idx=', idx)


    def pre_predict(self):
        for idx, sprite in enumerate(self.img_generator(self.img)):
            self.img_label[idx] = self.default_predictor(sprite)

    def tag(self):
        self.pre_predict()

        font = cv2.FONT_HERSHEY_SIMPLEX
        param = {'self':self}
        cv2.namedWindow('Tagging')
        cv2.setMouseCallback('Tagging', Tagging.mouse_event, param)

        while (1):
            tag_image = np.zeros((640 + 100, 640, 3), np.uint8)
            for idx in range(8 * 8):
                sprite = cv2.resize(self.img_array[(idx*32):(idx*32+32), :, :], (64, 64), cv2.INTER_CUBIC)
                _x, _y = int(idx / 8), int(idx % 8)
                if self.ctrl_mode == 0 or self.img_label[idx] == self.crt_label:
                    tag_image[(_x * 80 + 8):(_x * 80 + 72), (_y * 80 + 8):(_y * 80 + 72)] = sprite
                if self.img_label[idx] == self.crt_label and self.ctrl_mode == 0:
                    cv2.rectangle(tag_image, (_y * 80 + 8, _x * 80 + 8), (_y * 80 + 72, _x * 80 + 72), (0, 255, 0), 2)

            cv2.putText(tag_image, 'Current label = %s' % self.crt_label, (10, 680), font, 1.0, (0, 255, 0), 2)
            cv2.imshow('Tagging', tag_image)
            key = cv2.waitKey(200) & 0xFF
            if (key >= ord('0') and key <= ord('9')):
                self.crt_label = key - ord('0')
                print('key=', key, 'chr(key)=', chr(key))
            if key == 27:
                break

        print('Saving tag results ... ', )
        cv2.destroyWindow('Tagging')

        return self.save()

    def save(self):
        t = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        cv2.imwrite('img_data/'+t+'-data.jpg', self.img_array)
        np.save('img_data/'+t+'-label.npy', self.img_label)
        print('Done! Saved to img_data/'+t+'-...')
        return t

    def load(self, t):
        self.img_array = cv2.imread('img_data/'+t+'-data.jpg')
        self.img_label = np.load('img_data/'+t+'-label.npy')

if __name__ == '__main__':
    img = cv2.imread('video/Gem.jpg')[45:661, 313:929]
    tag = Tagging(img)
    t = tag.tag()
    tag.load(t)
    cv2.imshow('reloader', tag.img_array)
    cv2.waitKey(3000)
    cv2.destroyAllWindows()

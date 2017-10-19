#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File ROISelector.py created on 18:38 2017/9/11 

@author: Yichi Xiao
@version: 1.0
"""

import numpy as np
import cv2

x1, x2, y1, y2 = 0, 0, 0, 0

def mouse_select(event, x, y, flags, param):
    global x1, y1, x2, y2
    if event == cv2.EVENT_LBUTTONDOWN:
        x1 = x
        y1 = y
    if event == cv2.EVENT_LBUTTONUP or flags & cv2.EVENT_FLAG_LBUTTON:
        x2 = x
        y2 = y


def selectROI(image, ratio=None, round8=True):
    global x1, x2, y1, y2
    if ratio:
        a, b, _ = image.shape
        y1, y2 = round(a*ratio[1]), round(a*ratio[3])
        x1, x2 = round(b*ratio[0]), round(b*ratio[2])

        return image[y1:y2, x1:x2, :]

    cv2.imshow('ROI Selector', image)
    cv2.setMouseCallback('ROI Selector', mouse_select)
    x1, y1 = 0, 0
    y2, x2, _ = image.shape
    img = np.array(image, copy=True)
    while(1):
        img[:] = image
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.imshow('ROI Selector', img)
        if (cv2.waitKey(200) & 0xFF) == 27:
            break
    if round8:
        axis0 = round((x2-x1)/8)*8
        axis1 = round((y2-y1)/8)*8
        y1 = round((y1+y2)/2 - axis0/2)
        y2 = round((y1+y2)/2 + axis0/2)
        x1 = round((x1+x2)/2 - axis1/2)
        x2 = round((x1+x2)/2 + axis1/2)
    print("[ROI] img.shape = (%s, %s, %s)" % image.shape)
    print("[ROI] roi = img[%s:%s, %s:%s, :]"%(y1, y2, x1, x2))
    print("[ROI] width=%s(%s), height=%s(%s)"%(x2-x1, (x2-x1)/8, y2-y1, (y2-y1)/8))
    cv2.imshow('ROI Selector', image[y1:y2, x1:x2])
    print("[ROI] Y=(%.4f, %.4f), X=(%.4f, %.4f)" % (y1/image.shape[0], y2/image.shape[0],x1/image.shape[1],x2/image.shape[1]))

    cv2.waitKey()
    if (y2-y1)*(x2-x1) == 0:
        print("[WARNING] selectROI returns empty image. ignore it.")
        return image
    return image[y1:y2, x1:x2, :]


from BejeweledEnvironment import *

def new_main():
    env = BejeweledEnvironment(ratio=1.25)
    act = BejeweledAction()

    time.sleep(5)
    tick = 0
    ts = 0
    predictions, digits = env.get_initial_state()
    while True:
        tick += 1
        te = time.time()
        duration = int((te - ts) * 1000)
        print(tick, duration, 'ms! ', )
        ts = time.time()
        # predictions, digits = next(env.state_iterator)

        if tick >= 20 and tick % 1 == 0:
            action = act.random_action()
            predictions, digits = env.step(action)

        env.render()

        if np.count_nonzero(predictions == 0) > 40:
            print("No detection, sleep for 3 seconds.")
            time.sleep(3)

        if (tick > 0 and tick % 200000 == 0) or cv2.waitKey(10) & 0xFF == ord(' '):
            print('Learning mode!')
            tag = Tagging(env.last_image)
            tag.tag()

        print(tick)

if __name__ == '__main__':
    new_main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File ROISelector.py created on 18:38 2017/9/11 

@author: Yichi Xiao
@version: 1.0
"""

import numpy as np
import cv2
from Tagging import Tagging

x1, x2, y1, y2 = 0, 0, 0, 0

def mouse_select(event, x, y, flags, param):
    global x1, y1, x2, y2
    if event == cv2.EVENT_LBUTTONDOWN:
        x1 = x
        y1 = y
    if event == cv2.EVENT_LBUTTONUP or flags & cv2.EVENT_FLAG_LBUTTON:
        x2 = x
        y2 = y


def selectROI(image):
    global x1, x2, y1, y2
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
    print("img[%s:%s, %s:%s]"%(y1, y2, x1, x2))
    print("width=%s(%s), height=%s(%s)"%(x2-x1, (x2-x1)/8, y2-y1, (y2-y1)/8))
    cv2.imshow('ROI Selector', image[y1:y2, x1:x2])
    print("Y=(%.4f, %.4f), X=(%.4f, %.4f)" % (y1/image.shape[0], y2/image.shape[0],x1/image.shape[1],x2/image.shape[1]))

    cv2.waitKey()
    return image[y1:y2, x1:x2, :]

## From Picture
if False:
    img = cv2.imread('video/Gem.jpg')
    selectROI(img)
    img = img[42:664, 314:924]

## From Video (same as above)
if False:
    cap = cv2.VideoCapture("F:\\Workspace\\OpencvStarter\\video\\Gem.flv")
    tick = 0
    while(1):
        ret, img = cap.read()
        print(tick)
        tick+=1
        if not ret:
            break

        result = selectROI(img)
        cv2.imshow('Tool', result)

## From ImageGrab
from GrabSreen import *
tick = 0
hwnd = getHwnd("Bejeweled 3", "MainWindow")
te, ts = 0, 0

# conv net
# from cnn_sprite import SpriteConvnetModel
from img_utils import img_crop_to_array
print("Loading Tensorflow ...")
# model = SpriteConvnetModel()

while(1):
    duration = int((te-ts)*1000)
    print(tick, duration, 'ms')
    ts = time.time()
    tick+=1
    img = grabScreen(hwnd, delay=0.1, forceFront=(tick<10))
    if img is not None:
        pass
    else:
        print("Error! sleep for 3 seconds")
        time.sleep(3)
        continue
    result = selectROI(img)
    result = np.array(img[74:738, 342:998], copy=True)

    # predict via conv net
    if False:
        data = img_crop_to_array((result))
        predictions = model.predict(data)
        print(predictions)

        result = Tagging.attach(result, predictions)

    cv2.putText(result, '%s ms' % duration, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3)
    cv2.imshow('ROI', result)
    te = time.time()
    if (tick>0 and tick%200==0) or cv2.waitKey(10) & 0xFF == ord(' '):
        print('Learning mode!')
        tag = Tagging(img[74:738, 342:998])
        tag.tag()

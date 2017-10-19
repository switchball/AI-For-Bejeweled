#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File ROISelector.py created on 18:38 2017/9/11 

@author: Yichi Xiao
@version: 1.0
"""

import numpy as np
import cv2
from GrabSreen import *
from SpriteConvnetModel import SpriteConvnetModel, tf_flags
from img_utils import img_crop_to_array
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


def selectROI(image, ratio=None, round8=True):
    global x1, x2, y1, y2
    if ratio:
        a, b, _ = image.shape
        y1, y2 = round(a*ratio[0]), round(a*ratio[1])
        x1, x2 = round(b*ratio[2]), round(b*ratio[3])
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


def gen_roi_image():
    ## From ImageGrab
    score = 0
    tick = 0
    hwnd = getHwnd("Bejeweled 3", "MainWindow")

    while(1):
        tick+=1
        img = grabScreen(hwnd, delay=0.01, forceFront=(tick<10))
        if img is not None:
            pass
        else:
            print("Error! sleep for 3 seconds")
            time.sleep(3)
            continue
        result = selectROI(img, ratio=(0.1175, 0.9088, 0.3305, 0.9572))

        # 临时测试代码开始

        import pyocr
        import pyocr.tesseract as tess
        import pyocr.builders
        from PIL import Image

        digit_ratio = (0.1870, 0.2195, 0.1016, 0.2228)
        digits = selectROI(img, ratio=digit_ratio, round8=False)
        bw_img = digits
        # (thresh, bw_img) = cv2.threshold(digits, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        cv2.imwrite('digit_sample.jpg', bw_img)
        txt = tess.image_to_string(
            Image.fromarray(bw_img),
            lang='eng',
            builder=pyocr.builders.TextBuilder()
        )
        txt = txt.replace(',','').replace('.','')
        last_score = score
        if txt.isdigit():
            score = int(txt)
        else:
            score = last_score
        print(score, ' +', score - last_score)

        # 临时测试代码结束

        # it is the double yield trick, not bug
        yield result
        yield result




def gen_image_features(gen_IMG):
    for image_roi in gen_IMG:
        yield img_crop_to_array(image_roi)

from itertools import *
roi_img_generator = gen_roi_image()
model = SpriteConvnetModel(tf_flags(), False, True)
gen2 = model.predictor(gen_image_features(roi_img_generator))
#gen2 = model.predictor(starmap(img_crop_to_array, roi_img_generator))

tick = 0
te, ts = 0, 0
for roi_img in roi_img_generator:
    te = time.time()
    print('='*20)
    duration = int((te - ts) * 1000)
    print(tick, duration, 'ms!')
    ts = time.time()
    tick += 1

    predictions = gen2.__next__()
    result = Tagging.attach(roi_img.copy(), predictions)

    cv2.putText(result, '%s ms' % duration, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3)
    cv2.imshow('Sprites', result)
    if (tick > 0 and tick % 2000 == 0) or cv2.waitKey(10) & 0xFF == ord(' '):
        print('Learning mode!')
        tag = Tagging(roi_img)
        tag.tag()




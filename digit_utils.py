#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File digit_utils.py created on 22:39 2017/10/18 

@author: Yichi Xiao
@version: 1.0
"""

import numpy as np
import cv2

from PIL import Image
import sys

import pyocr.tesseract as tess
import pyocr.builders

import time
import os, subprocess



ts = time.time()
txt = tess.image_to_string(
    Image.open('digit_sample.jpg'),
    lang='eng',
    builder=pyocr.builders.TextBuilder()
)
#print(txt)
#print(time.time() - ts)
# txt is a Python string

def image_to_string(img, cleanup=True, plus=''):
    # cleanup为True则识别完成后删除生成的文本文件
    # plus参数为给tesseract的附加高级参数
    # subprocess.check_output('tesseract ' + img + ' ' +
    #                         img + ' ' + plus, shell=True)  # 生成同名txt文件
    os.popen('tesseract ' + img + ' ' +
                            img + ' ' + plus)
    text = ''
    with open(img + '.txt', 'r') as f:
        text = f.read().strip()
    if cleanup:
        os.remove(img + '.txt')
    return text

ts = time.time()
txt = image_to_string('digit_sample.jpg', True, '-l eng')
print(txt)
print(time.time() - ts)
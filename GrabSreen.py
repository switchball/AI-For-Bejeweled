#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File GrabSreen.py created on 15:33 2017/9/11 

@author: Yichi Xiao
@version: 1.0
"""

import numpy as np
import cv2
import time

import win32gui
from PIL import ImageGrab
import win32con


def getHwnd(caption, clazz):
    hwnd =  win32gui.FindWindow(clazz, caption)
    if not hwnd:
        print('window not found!')
    return hwnd


def grabScreen(hwnd, delay=0.25, forceFront=False, ratio=1.25):
    try:
        if forceFront:
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)  # 强行显示界面后才好截图
            win32gui.SetForegroundWindow(hwnd)  # 将窗口提到最前
    except:
        return None
    time.sleep(delay)
    #  裁剪得到全图
    game_rect = win32gui.GetWindowRect(hwnd)
    game_rect = tuple(int(1.25*x) for x in game_rect)
    pil_image = ImageGrab.grab(game_rect).convert('RGB')
    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_BGR2RGB)
    return image


if __name__ == '__main__':

    caption = "Bejeweled 3"
    clazz = "MainWindow"
    hwnd = win32gui.FindWindow(clazz, caption)
    if not hwnd:
        print('window not found!')
    else:
        print(hwnd)

    win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)  # 强行显示界面后才好截图
    win32gui.SetForegroundWindow(hwnd)  # 将窗口提到最前
    time.sleep(1)
    #  裁剪得到全图
    game_rect = win32gui.GetWindowRect(hwnd)
    game_rect_2 = tuple(int(1.25*x) for x in game_rect)
    print(game_rect)
    print(game_rect_2)
    pil_image = ImageGrab.grab(game_rect_2)
    #pil_image = ImageGrab.grab((game_rect[0] + 9, game_rect[1] + 190, game_rect[2] - 9, game_rect[1] + 190 + 450))
    image = np.array(pil_image.getdata(),dtype='uint8')\
        .reshape((pil_image.size[1],pil_image.size[0],3))

    cv2.imshow("Screen", image)
    cv2.waitKey()
    cv2.destroyAllWindows()
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File BejeweledEnvironment.py created on 15:39 2017/10/19 

@author: Yichi Xiao
@version: 1.0
"""

import numpy as np
import cv2
import time
import win32api
import win32gui
from PIL import ImageGrab
import win32con

from itertools import product

from Environment import Environment

def interpolate(a, b, t):
    if t <= 0:
        return a
    if t >= 1:
        return b
    return a + (b-a) * t

class BejeweledAction():
    def __init__(self):
        self.action_space = list(product([0,1,2,3,4,5,6,7], [0,1,2,3,4,5,6], ['H','V']))

    def random_action(self):
        return self.action_space[np.random.randint(len(self.action_space))]

class BejeweledEnvironment(Environment):
    def __init__(self, ratio=1.25):
        super(BejeweledEnvironment, self).__init__()
        self.ratio = ratio
        self.SPRITE_RATIO = (0.3305, 0.1175, 0.9572, 0.9088)
        self.hwnd = None
        self.game_rect = None
        self.getHwnd("Bejeweled 3", "MainWindow")
        self.getScreenResolution()

    def step(self, action):
        a, b, c = action
        if c == 'H':
            row1, row2 = a, a
            col1, col2 = b, b+1
        elif c == 'V':
            col1, col2 = a, a
            row1, row2 = b, b+1
        else:
            print("Invalid Action!")
            return None
        print("Step Action:", action)
        self.mouse_click_on_sprite(row1, col1)
        time.sleep(0.05)
        self.mouse_click_on_sprite(row2, col2)


    def getScreenResolution(self):
        width = win32api.GetSystemMetrics(0)
        height = win32api.GetSystemMetrics(1)
        self.screenSize = (width, height)

    def getHwnd(self, caption, clazz):
        self.hwnd = win32gui.FindWindow(clazz, caption)
        if not self.hwnd:
            print('window not found!')
        return self.hwnd

    def grabScreen(self, delay=0.25, forceFront=False):
        try:
            if forceFront:
                win32gui.ShowWindow(self.hwnd, win32con.SW_RESTORE)  # 强行显示界面后才好截图
                win32gui.SetForegroundWindow(self.hwnd)  # 将窗口提到最前
        except:
            return None
        time.sleep(delay)
        #  裁剪得到全图
        game_rect = win32gui.GetWindowRect(self.hwnd)
        game_rect = tuple(int(self.ratio * x) for x in game_rect)
        self.game_rect = game_rect
        pil_image = ImageGrab.grab(game_rect).convert('RGB')
        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_BGR2RGB)

        r = self.getCursorRatio()
        self.gameRatioToScreenPoint(r)
        return image

    def getCursorRatio(self):
        cursor_info = win32gui.GetCursorInfo()[2]
        c = (cursor_info[0]*1.25, cursor_info[1]*1.25)
        gr = self.game_rect
        # print("cursor: ", cursor_info)
        # print("cursor: ", (cursor_info[0]*1.25, cursor_info[1]*1.25))
        # print("game_rect:", self.game_rect)
        r1 = 1.0 * (c[0] - gr[0]) / (gr[2] - gr[0])
        r2 = 1.0 * (c[1] - gr[1]) / (gr[3] - gr[1])
        # print("cursor ratio:", (r1, r2))
        return (r1, r2)

    def gameRatioToScreenPoint(self, cur_ratio):
        ''' input: (ratio_w, ratio_h)
            pos_w = game_rect_ws + game_rect_width * ratio_w
            pos_h = game_rect_hs + game_rect_height* ratio_h
        '''
        gr = self.game_rect
        pos_w = gr[0] * (1-cur_ratio[0]) + gr[2] * cur_ratio[0]
        pos_h = gr[1] * (1-cur_ratio[1]) + gr[3] * cur_ratio[1]
        p1 = int(pos_w/self.ratio)
        p2 = int(pos_h/self.ratio)
        # print("gameRatioToScreenPoint:", (p1, p2))
        return (p1, p2)

    # def gamePointToScreenPoint(self):
    #     pass

    def mouse_click_on_sprite(self, row, col):
        r1 = interpolate(self.SPRITE_RATIO[0], self.SPRITE_RATIO[2], (row+0.5)/8)
        r2 = interpolate(self.SPRITE_RATIO[1], self.SPRITE_RATIO[3], (col+0.5)/8)
        pos = self.gameRatioToScreenPoint((r1, r2))
        self.mouse_click_on_screen(pos)
        pass

    def mouse_click_on_screen(self, pos):
        win32api.SetCursorPos(pos)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN | win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)

    def send_esc(self):
        win32api.keybd_event(27, 0, 0, 0)
        win32api.keybd_event(27, 0, win32con.KEYEVENTF_KEYUP, 0)
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File Environment.py created on 15:25 2017/10/19 

@author: Yichi Xiao
@version: 1.0
"""


class Environment(object):
    def __init__(self):
        self.observation_space = None
        self.action_space = None

    def reset(self):
        pass

    def step(self, action):
        ''' :return next_state, reward, done, _ '''
        pass

    def render(self):
        pass
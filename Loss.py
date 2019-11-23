#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 19:25:34 2019
Loss Functions
@author: diepy
"""
import numpy as np

# loss
def cross_entropy(actual, predicted):
    out_num = actual.shape[0]
    p = np.log(predicted)
    loss = -np.sum(actual.reshape(1,out_num)*p)
    return loss

def delta_loss(actual, predicted):
    return -actual/predicted
    
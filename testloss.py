#!/usr/bin/env python
# -*- coding: utf-8 -*-
# (c) 2019 gmrandazzo@gmail.com
# This file is part of DeepDioEq.
# You can use,modify, and distribute it under
# the terms of the GNU General Public Licenze, version 3.
# See the file LICENSE for details

import numpy as np
import tensorflow as tf
from train import floss

def cube(x):
    return x*x*x

def loss(y_true, y_false):
    int_true = cube(y_true[:,0])
    int_true += cube(y_true[:,1])
    int_true += cube(y_true[:,2])
    int_pred = cube(y_pred[:,0])
    int_pred += cube(y_pred[:,1])
    int_pred += cube(y_pred[:,2])
    res = np.square(int_true - int_pred)
    res = np.sqrt(res)
    res = np.mean(res)
    return res


if __name__ in "__main__":
    y_true = []
    y_true.append([0,0,1])
    y_true.append([-1, -1, 2])

    y_pred = []
    y_pred.append([0, 0, 1.5])
    y_pred.append([-1, -1, 2])
    y_true = np.array(y_true).astype(float)
    y_pred = np.array(y_pred).astype(float)
    nploss = loss(y_true, y_pred)
    sess = tf.compat.v1.Session()
    tfloss = floss(y_true, y_pred).eval(session=sess)
    print("%f == %f" % (nploss, tfloss))

    



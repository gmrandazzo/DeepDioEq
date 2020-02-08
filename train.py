#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# (c) 2019 gmrandazzo@gmail.com
# This file is part of DeepDioEq.
# You can use,modify, and distribute it under
# the terms of the GNU General Public Licenze, version 3.
# See the file LICENSE for details

import sys
import time
import keras
from keras import backend as K
from keras.models import Sequential
from keras.models import Model
from keras.models import load_model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
import tensorflow as tf
import numpy as np
import random

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split as tss
from sklearn.metrics import r2_score as r2
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae

from pathlib import Path

def make_number(xyz):
    def cb(x):
        return x*x*x
    return cb(xyz[0]) + cb(xyz[1]) + cb(xyz[2])

@tf.function
def cube(x):
    return x*x*x

def floss(y_true, y_pred):
    # How far we are from the solution...
    #y_true = tf.math.ceil(y_true)
    #y_true = tf.Print(y_true, [y_true], "True: ")
    #y_pred = tf.Print(y_pred, [y_pred], "Pred: ")
    int_true = cube(y_true[:,0])
    int_true += cube(y_true[:,1])
    int_true += cube(y_true[:,2])
    int_pred = cube(y_pred[:,0])
    int_pred += cube(y_pred[:,1])
    int_pred += cube(y_pred[:,2])
    #int_true = tf.print(int_true)
    #int_pred = tf.print(int_pred)
    #int_true = tf.Print(int_true, [int_true], "True: ")
    #int_pred = tf.Print(int_pred, [int_pred], "Pred: ")
    #res = tf.abs(int_true - int_pred)
    res_n = K.mean(K.square(int_true - int_pred), axis=-1)
    #res_n = tf.square(int_true - int_pred)
    #res_n = tf.reduce_mean(res_n)
    res_xyz = K.mean(K.square(y_pred - y_true), axis=-1)
    return K.mean(res_n+res_xyz)

def rsq(y_true, y_pred):
    int_true = cube(y_true[:,0])
    int_true += cube(y_true[:,1])
    int_true += cube(y_true[:,2])
    int_pred = cube(y_pred[:,0])
    int_pred += cube(y_pred[:,1])
    int_pred += cube(y_pred[:,2])
    SS_res =  K.sum(K.square(int_true - int_pred)) 
    SS_tot = K.sum(K.square(int_true - K.mean(int_pred))) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def build_model(nunits, ndense):
    m = Sequential()
    """
    m.add(BatchNormalization(input_shape=(1,)))
    m.add(Dense(nunits, activation='relu'))
    """
    m.add(Dense(nunits,
                input_shape=(1,),
                activation='relu'))
    for i in range(ndense):
        m.add(Dense(nunits,
                    activation='relu'))
    m.add(Dense(3))
    m.compile(loss=floss,
              optimizer=optimizers.Adam(lr=0.0005),
              metrics = [rsq, floss, 'mse','mae'])
    return m

def makeSample(x, y, ids):
    x_sample = []
    y_sample = []
    for i in ids:
        x_sample.append(x[i])
        y_sample.append(y[i])
    return np.array(x_sample).astype(float), np.array(y_sample).astype(float)

    
def train_test_split(x, y, th=0.2, random_state=None):
    ids = list(x.keys())
    train, val = tss(ids, random_state=random_state)
    x_train, y_train = makeSample(x, y, train)
    x_test, y_test = makeSample(x,y, val)
    return x_train, y_train, x_test, y_test

def validation(x, y, nsplits=5, nrepeats=4):
    ids = np.array(list(x.keys()))
    for i in range(nrepeats):
        kf = KFold(n_splits=nsplits)
        for subset_indx, test_indx in kf.split(ids):
            train_keys, val_keys = tss(ids[subset_indx])
            x_train, y_train =  makeSample(x, y, train_keys)
            x_val, y_val = makeSample(x,y, val_keys)
            x_test, y_test = makeSample(x, y, ids[test_indx])
            yield x_train, y_train, x_val, y_val, x_test, y_test, train_keys, val_keys, ids[test_indx]
            
class NN(object):
    def __init__(self, csv_tab):
        self.X, self.y = self.ReadTable(csv_tab)
   
    def ReadTable(self, csv_tab):
        x = {}
        y = {}
        f =open(csv_tab, 'r')
        indx = 0
        for line in f:
            if "n" in line:
                continue
            else:
                v = str.split(line.strip(), ',')
                x["num%d" % indx] = v[0]
                y["num%d" % indx] = v[1:]
                indx += 1
        f.close()
        return x, y

    def makeModel(self,
                  x_train,
                  y_train,
                  x_val,
                  y_val,
                  nunits,
                  ndense,
                  epochs,
                  batch_size,
                  mout_file=None):
        print("# train %d  # val %d" % (x_train.shape[0],
                                        x_val.shape[0]))
        log_dir_ = "./logs/%s" % (time.strftime("%Y%m%d%H%M%S"))
        log_dir_ += "_#u%d_#dl%d_#epochs%d_#batchsize%d" % (nunits, ndense, epochs, batch_size)
        callbacks_ = None
        if mout_file is not None:
            callbacks_=[TensorBoard(log_dir=log_dir_,
                        histogram_freq=0,
                        write_graph=False,
                        write_images=False),
                        ModelCheckpoint(mout_file,
                                       monitor="val_loss",
                                       verbose=0,
                                       save_best_only=True)]
        else:
            callbacks_=[TensorBoard(log_dir=log_dir_,
                                    histogram_freq=0,
                                    write_graph=False,
                                    write_images=False)]

        m = build_model(nunits, ndense)
        m.fit(x_train,
              y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=0,
              validation_data=(x_val, y_val),
              callbacks=callbacks_)
        if mout_file is not None:
            m = load_model(mout_file, custom_objects={ "rsq": rsq, "floss":floss})
        return m
    
    def makePrediction(self, model, x):
        return model.predict(x)

    def simplerun(self,
                  nunits,
                  ndense,
                  epochs,
                  batch_size):
        x_train, y_train, x_val, y_val = train_test_split(self.X,
                                                          self.y,
                                                          0.2,
                                                          123456789)
        
        m = self.makeModel(x_train,
                           y_train,
                           x_val,
                           y_val,
                           nunits,
                           ndense,
                           epochs,
                           batch_size,
                           'tmpmodel.h5')
        
        y_pred = self.makePrediction(m, x_val)
        x = []
        y = []
        for i in range(len(y_pred)):
            x.append(make_number(y_val[i]))
            y.append(make_number(y_pred[i]))
        print("R2: %.4f MSE: %.2f MAE :%.2f" % (r2(x,y),
                                                mse(x,y),
                                                mae(x,y)))
        for i in range(len(y_val)):
            s = "%d:" % (i)
            for j in range(len(y_val[i])):
                s += "\t%f %f" % (y_val[i][j], y_pred[i][j])
            print(s)
        
        return 0 

    def cv(self,
           nunits,
           ndense,
           epochs,
           batch_size,
           nsplits,
           nrepeats,
           mout_path):
        cv = 0
        
        predictions = {}
        for key in self.X.keys():
            predictions[key] = []
        
        mpath = Path(mout_path)
        mpath.mkdir(exist_ok=True)
        for x_train, y_train, x_val, y_val, x_test, y_test, train_keys, val_keys, test_keys in validation(self.X, self.y, nsplits, nrepeats):
            mout_file="%s/%d.h5" % (mpath, cv)
            m = self.makeModel(x_train,
                               y_train,
                               x_val,
                               y_val,
                               nunits,
                               ndense,
                               epochs,
                               batch_size,
                               mout_file)

            y_pred = self.makePrediction(m, x_test)
            x = []
            y = []
            for i in range(len(test_keys)):
                x.append(make_number(y_test[i]))
                y.append(make_number(y_pred[i]))
                predictions[test_keys[i]].append(float(y[-1]))
                predictions[test_keys[i]].extend(y_pred[i])
                
            print("R2: %.4f MSE: %.2f MAE :%.2f" % (r2(x,y),
                                                    mse(x,y),
                                                    mae(x,y)))
            cv += 1
        fo = open("cv_%s.csv" % (mout_path), "w")
        for key in predictions.keys():
            fo.write("%s,%s,%s,%s,%s" % (key,
                                         self.X[key],
                                         self.y[key][0],
                                         self.y[key][1],
                                         self.y[key][2]))
            for item in predictions[key]:
                fo.write(",%f" % (item))
            fo.write("\n")
        fo.close()
        return 0

def main():
    print("DeepDio(phantine)Eq(ations) solver")
    print("\nThis is an attempt to solve the")
    print("sum of three cubes problem")
    print("problem also known to be a Diophantine equation")
    
    if len(sys.argv) == 5:
        epochs = int(sys.argv[1])
        batch_size = int(sys.argv[2])
        nunits = int(sys.argv[3])
        ndense = int(sys.argv[4])
        nn = NN("dataset.csv")
        nn.simplerun(nunits, ndense, epochs, batch_size)
    elif len(sys.argv) == 8:
        epochs = int(sys.argv[1])
        batch_size = int(sys.argv[2])
        nunits = int(sys.argv[3])
        ndense = int(sys.argv[4])
        nsplits = int(sys.argv[5])
        nrepeats = int(sys.argv[6])
        mout_path = sys.argv[7]
        nn = NN("dataset.csv")
        nn.cv(nunits, ndense, epochs, batch_size, nsplits, nrepeats, mout_path)
        
    else:
        print("\nUsage %s [epochs] [batch_size] [nunits] [ndense layers]" % sys.argv[0])
        print("\nUsage %s [epochs] [batch_size] [nunits] [ndense layers] [nsplits] [nrepeats] [model out]" % sys.argv[0])

    return

if __name__ in "__main__":
    main()



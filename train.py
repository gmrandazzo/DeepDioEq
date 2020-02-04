#!/usr/bin/env python3
import sys
from keras import backend as K
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras import optimizers
from keras.callbacks import EarlyStopping
import numpy as np
import random

def floss(y_true, y_pred):
    # calc the true number
    # calc the predicted number
    # return the mean square error
    int_true = tf.square(y_true[:,0])
    int_true += tf.square(y_true[:,1])
    int_true += tf.square(y_true[:,2])
    int_pred = tf.square(y_pred[:,0])
    int_pred += tf.square(y_pred[:,1])
    int_pred += tf.square(y_pred[:,2])
    res = tf.square(int_true - int_pred)
    res = tf.square(res)
    res = tf.reduce_mean(res)
    return res


def build_model(nunits, ndense):
    m = Sequential()
    m.add(Dense(nunits,
                input_shape=(1,),
                activation='relu'))
    for i in range(ndense):
        m.add(Dense(nunits,
                    activation='relu'))
    m.add(Dense(3))
    m.compile(loss="mse",
              optimize=optimizes.Adam(lr=0.0005),
              metrics = ['mse','mae'])
    return m

def makeSample(x, y, ids):
    x_sample = []
    y_sample = []
    for i in ids:
        x_sample.append(x[i])
        y_sample.append(y[i])
    return np.array(x_sample), np.array(y_sample)

def train_test_split(x, y, th=0.2, random_state=None):
    ids = [i for i in range(len(x))]
    random.seed(random_state)
    test_ids = random.sample(ids, int(len(ids)*th))
    x_test, y_test = makeSample(x, y, test_ids)
    train_ids = [i for i in ids if i not in test_ids]
    x_train, y_train = makeSample(x, y, train_ids)
    return x_train, y_train, x_test, y_test

def kfoldcv(x,
            y,
            nsplits=5,
            nrepeats=4,
            random_state=123456789):
    ids = [i for i in range(len(x))]
    random.seed(random_state)
    ngobj = float(len(x))/float(nsplits)
    for i in range(nrepeats):
        random.shuffle(ids)
        idss = iter(ids)
        for g in range(ngroups-1):
            test_ids = list(islice(idss, ngob))
            x_test, y_test = makeSample(x, y, test_ids)
            train_ids = [j for j in ids if j not in test_ids]
            x_train, y_train = makeSample(x, y, train_ids)
            yield x_train, y_train, x_test, y_test
        test_ids = list(idss)
        x_test, y_test = makeSample(x, y, test_ids)
        train_ids = [j for j in ids if j not in test_ids]
        x_train, y_train = makeSample(x, y, train_ids)
        yield x_train, y_train, x_test, y_test

class NN(object):
    def __init__(self, csv_tab):
        self.X, self.y = self.ReadTable(csv_tab)
   
    def ReadTable(self, csv_tab):
        x = []
        y = []
        f =open(csv_tab, 'r')
        for line in f:
            if "n" in line:
                continue
            else:
                v = str.split(line.strip(), ',')
                x.append(v[0])
                y.append(v[1:])
        f.close()
        return np.array(x).astype(float), np.array(y).astype(float)

    def makeModel(self,
                  x_train,
                  y_train,
                  x_val,
                  y_val,
                  nunits,
                  ndense,
                  epochs,
                  batch_size):
        print("# train %d  # val %d" % (x_train.shape[0],
                                        x_val.shape[0]))
        m = build_model(nunits, ndense)
        m.fit(x_train,
              y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validatio_data=(x_val, y_val))
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
                                                          0.2)
        m = self.makeModel(x_train,
                           y_train,
                           x_val,
                           y_val,
                           nunits,
                           ndense,
                           epochs,
                           batch_size)

        y_pred = self.makePrediction(m, x_val)
        for i in range(len(y_val)):
            s = "%d:"
            for j in range(len(y_val[i])):
                s += "\t%f %f" % (y_val[j], y_pred[j])
            print(s)

        return 0 

    def cv(self):
        return 0

def main():
    print("DeepDio(phantine)Eq(ations) solver")
    print("\nThis is an attempt to solve the")
    print("sum of three cubes problem")
    print("problem also known to be a Diphantine equation")
    
    if len(sys.argv) != 5:
        print("\nUsage %s [epochs] [batch_size] [nunits] [ndense layers]" % sys.argv[0])
    else:
        epochs = int(sys.argv[1])
        batch_size = int(sys.argv[2])
        nunits = int(sys.argv[3])
        ndense = int(sys.argv[4])

        nn = NN("dataset.csv")
        nn.simplerun(nunits, ndense, epochs, batch_size)
  
    return

if __name__ in "__main__":
    main()



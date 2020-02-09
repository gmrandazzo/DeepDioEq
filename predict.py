#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# (c) 2019 gmrandazzo@gmail.com
# This file is part of DeepDioEq.
# You can use,modify, and distribute it under
# the terms of the GNU General Public Licenze, version 3.
# See the file LICENSE for details

import sys
from keras.models import load_model
from pathlib import Path
from train import floss
from train import rsq
from train import make_number

def ReadNumList(txtfile):
    f = open(txtfile, "r")
    n = []
    for line in f:
        n.append(int(line.strip()))
    f.close()
    return n

def LoadModels(model_path):
    models = []
    modelspath = Path(model_path).glob('**/*.h5')
    for path in modelspath:
        models.append(load_model(path, custom_objects={ "rsq": rsq, "floss":floss}))
    return models


def makePrediction(models, numlst):
    predictions = {}
    for n in numlst:
        predictions[n] = []
    
    for m in models:
        for n in numlst:
            y = m.predict([n])
            npred = make_number(y[-1])
            predictions[n].append(npred)
            predictions[n].extend(y[-1])
    return predictions

def WriteCSV(predictions, csvout):
    fo = open(csvout, "w")
    for key in predictions.keys():
        fo.write("%d" % (key))
        for item in predictions[key]:
            fo.write(",%f" % (item))
        fo.write("\n")
    fo.close()
    
def main():
    if len(sys.argv) != 4:
        print("\nUsage: %s [model path] [txt list of numbers to predict] [prediction output]" % (sys.argv[0]))
    else:
        models = LoadModels(sys.argv[1])
        nlst = ReadNumList(sys.argv[2])
        predictions = makePrediction(models, nlst)
        WriteCSV(predictions, sys.argv[3])
    return 0

if __name__ in "__main__":
    main()


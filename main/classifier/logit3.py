#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
#@author  Bin Hong
import numpy as np

import keras
from keras.layers import Activation, Dense, Dropout
from keras.models import Sequential
from keras import optimizers
import os, sys

local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..', "..")
sys.path.append(root)

from main.classifier.base_classifier import BaseClassifier
from main.classifier.logit2 import Logit2
from main.classifier.interval_acc import IntervalAcc

class Logit3(Logit2):
    def __init__(self, dim = 64, hs = 3, batch_size = 100, nb_epoch=30, dropout=0.5, verbose = 1):
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
        self.verbose = verbose
        self.dim = dim
        self.hs = hs
        self.dropout = dropout
        self.opt = optimizers.Adagrad(lr=4e-5)

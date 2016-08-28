#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
#@author  Bin Hong

import sys,os
import json
import numpy as np
import pandas as pd
import math
import random
import multiprocessing
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib # to dump model
from sklearn import preprocessing

local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..', '..')
sys.path.append(root)

import main.model.model_param_set as params_set
import main.base as base
import main.ta as ta

def sw_none(df, label="label5", name = "sample_weight"):
    df["sample_weight"] = 1
    return df


def sw_linear(df, label="label5", name = "sample_weight"):
    df["sample_weight"] = 1+ abs(df.label5-1)
    return df

def sw_linear2(df, label="label5", name = "sample_weight"):
    df["sample_weight"] = abs(df.label5-1) * 100
    return df

def sw_nlinear(df, label="label5", name = "sample_weight"):
    df["sample_weight"] = 1 - abs(df.label5-1)
    return df


def gap(label):
    tmp = abs(label -1)
    if tmp > 0.10:
        return 0.1
    return 1 + abs(label - 1)

def sw_gap(df, label="label5", name = "sample_weight"):
    df["sample_weight"] = df.apply(lambda row: gap(row["label5"]), axis=1)
    return df

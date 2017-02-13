#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
#@author  Bin Hong

"""

"""

import sys,os
import pandas as pd
from sklearn.externals import joblib # to dump model
local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..', '..')
sys.path.append(root)
sys.path.append(local_path)

import main.base as base

def work(confer):
    ta_file = confer.get_ta_file()
    



    return df_pred[["date", "sym", "open", "high", "low", "close", "pred"]].head(20)

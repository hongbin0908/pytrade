#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author hongbin@youzan.com

import os,sys
import talib
import numpy as np
import pandas as pd

local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..', '..')
sys.path.append(root)

from main.utils import time_me
import main.pandas_talib as pta
import main.base as base

import main.ta.ta_base1_ext4 as base1_ext4

def main(df):
    df = df.reset_index("date", drop = True)
    df = base1_ext4.main(df)
    ipt = os.path.join(root,  "main/model/feat_select_sp500Top50-base1_ext4")

    with open(ipt, "r") as f:
        for line in f.readlines():
            line = line.split()
            if "F" == line[1]:
                if line[0] in df:
                    del df[line[0]]

    return df

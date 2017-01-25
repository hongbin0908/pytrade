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


def main(df):
    dfSp500 = pd.read_csv("./constituents-financials.csv")
    dfSector = dfSp500[["Symbol", "Market Cap"]]
    dfSector = dfSector.rename(columns = {"Symbol":"sym", "Market Cap":"ta_markcap"})
    df = pd.merge(how = 'left', left = df, right = dfSector, left_on = "sym", right_on = "sym")
    return df


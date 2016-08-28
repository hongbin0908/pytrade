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


def sector(key, value, sector):
    if sector == value:
        return 1
    return 0

def main(df):
    dfSp500 = pd.read_csv("./constituents-financials.csv")
    dfSector = dfSp500[["Symbol", "Sector"]]
    dfSector = dfSector.rename(columns = {"Symbol":"sym", "Sector":"sector"})
    df = pd.merge(how = 'left', left = df, right = dfSector, left_on = "sym", right_on = "sym")
    dSector = dict(enumerate(dfSector["sector"].unique()))
    for key in dSector.keys():
        df["ta_sector_%d" %  key ] = df.apply(lambda row: sector(key, dSector[key], row["sector"]), axis=1)
    return df


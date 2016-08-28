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

import main.ta.ta_base1 as base1
import main.ta.ta_cdl as cdl

def main(df):
    df = base1.main(df)
    df = cdl.main(df)
    print df.shape
    return df


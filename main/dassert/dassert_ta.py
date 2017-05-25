#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# @author  Bin Hong

import sys
import os
import pandas as pd
from pandas.util.testing import assert_frame_equal

local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..', "..")
sys.path.append(root)

from main import base

def work(confer):
    ta1 = confer.get_ta_file()
    last_trade_date = confer.last_trade_date
    confer.last_trade_date = base.get_second_trade_date_local(confer.syms.get_name())
    ta2 = confer.get_ta_file()
    confer.last_trade_date = last_trade_date


    df1 = pd.read_pickle(ta1)
    df2 = pd.read_pickle(ta2)

    assert len(df1) == len(df2)
    assert_frame_equal(df1, df2)

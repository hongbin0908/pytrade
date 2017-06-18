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
    score1 = confer.get_score_file()
    last_trade_date = confer.last_trade_date
    confer.last_trade_date = base.get_second_trade_date_local(confer.syms.get_name())
    print(confer.last_trade_date)
    score2 = confer.get_score_file()
    confer.last_trade_date = last_trade_date

    df1 = pd.read_pickle(score1)
    df1 = df1[df1.date <= base.get_second_trade_date_local(confer.syms.get_name())]
    df2 = pd.read_pickle(score2)

    syms1 = df1.sym.unique()
    syms2 = df2.sym.unique()

    print(syms1, syms2)
    assert len(syms1) == len(syms2)
    df1.reset_index(drop=True, inplace=True)
    df2.reset_index(drop=True, inplace=True)
    assert len(df1) == len(df2)
    #assert_frame_equal(df1[base.get_feat_score_names(df1)],
    #                   df2[base.get_feat_score_names(df2)])

    for sym in syms1:
        print(sym)
        df_s_1 = df1[df1.sym == sym]
        df_s_1.reset_index(drop=True, inplace=True)
        df_s_1 = df_s_1.drop(df_s_1.tail(10).index)
        df_s_2 = df2[df2.sym == sym]
        df_s_2.reset_index(drop=True, inplace=True)
        df_s_2 = df_s_2.drop(df_s_2.tail(10).index)
        if sym == "ADI":
            print(df_s_1[["date", "score_rel_5_0"]])
            print(df_s_2[["date", "score_rel_5_0"]])
        assert_frame_equal(df_s_1[base.get_feat_score_names(df_s_1)],
                           df_s_2[base.get_feat_score_names(df_s_2)])


    #assert len(df1) == len(df2)

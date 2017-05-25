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

def get_df(yeod_dir, each):
    if each.endswith('csv'):
        df = pd.read_csv(os.path.join(yeod_dir, each))
        df['sym'] = each
        return df
    return None

def merge_yeod(yeod_dir):
    return pd.concat([get_df(yeod_dir, each) for each in os.listdir(yeod_dir)])

def work(confer):
    yeod_dir1 = confer.get_yeod_dir()
    last_trade_date = confer.last_trade_date
    confer.last_trade_date = base.get_second_trade_date_local(confer.syms.get_name())
    yeod_dir2 = confer.get_yeod_dir()

    #print("yeod_dir1: %s; yeod_dir2: %s" %  (yeod_dir1, yeod_dir2))
    pd1 = merge_yeod(yeod_dir1)
    pd1 = pd1[pd1.date <= base.get_second_trade_date_local(confer.syms.get_name())]
    print(pd1.shape)
    pd2 = merge_yeod(yeod_dir2)
    print(pd2.shape)

    syms1 = pd1.sym.unique()
    syms2 = pd2.sym.unique()

    assert len(syms1) == len(syms2)
    assert len(pd1) == len(pd2)

    pd1 = pd1.sort_values(["sym",'date'])
    pd2 = pd2.sort_values(["sym",'date'])
    for sym in syms1:
        if sym in set(['EXPE.csv']):
            continue
        print(sym)
        df1 = pd1[pd1.sym == sym]
        df2 = pd2[pd2.sym == sym]
        assert_frame_equal(df1[["date",'openo','higho','lowo', 'closeo','volume']], df2[['date', 'openo', 'higho', 'lowo', 'closeo', 'volume']])
    assert_frame_equal(pd1[["date",'openo','higho','lowo', 'closeo']], pd2[['date', 'openo', 'higho', 'lowo', 'closeo']])

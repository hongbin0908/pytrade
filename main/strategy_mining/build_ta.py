
#!/usr/bin/env python
# -*- coding: utf-8 -*-

#@author 
import os,sys
import talib
import pandas as pd
import numpy as np
local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..')
sys.path.append(root)
import strategy_mining.model_base as base
import ta


def _judge(df, window):
    df["close_shift"] = df["close"].shift(-1 * window)
    df["label" + str(window)] = df["close_shift"]/df["close"]
    return df 

def judge(df):
    df = _judge(df, 1)
    df = _judge(df, 2)
    df = _judge(df, 3)
    df = _judge(df, 4)
    df = _judge(df, 5)
    df = _judge(df, 6)
    df = _judge(df, 8)
    df = _judge(df, 10)
    df = _judge(df, 20)
    df = _judge(df, 30)
    df = _judge(df, 60)
    return df
def get_pd(symbol):
    names = ["date", 'open', 'high', 'low', 'close', 'volume', 'adjclose']
    df = pd.read_csv(os.path.join(local_path, '..', 'data', 'yeod', symbol+".csv"), \
            header=None, names=names, \
            dtype={"volume":np.float64}, \
            skiprows=1, index_col='date', parse_dates=True).sort_index()
    return df

def main1():
    for each in base.get_file_list(os.path.join(local_path, '..', 'data', 'yeod')):
        symbol = base.get_stock_from_path(each)
        df = get_pd(base.get_stock_from_path(each))
        df = ta.cal_all(df)
        df = judge(df)
        df.to_csv(os.path.join(root, 'data1', 'ta', symbol + ".csv"))
def main2():
    for each in base.get_file_list(os.path.join(local_path, '..', 'data', 'yeod')):
        symbol = base.get_stock_from_path(each)
        df = get_pd(base.get_stock_from_path(each))
        df = ta.cal2(df)
        df = judge(df)
        df.to_csv(os.path.join(root, 'data2', 'ta', symbol + ".csv"))
if __name__ == '__main__':
    main1()
    main2()
#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

#@author Bin Hong
import os,sys
import talib
import pandas as pd
import numpy as np
import multiprocessing
local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..', '..')
sys.path.append(root)

import main.ta as ta

def get_file_list(rootdir):
    file_list = []
    for f in os.listdir(rootdir):
        if f == None or not f.endswith(".csv"):
            continue
        file_list.append(os.path.join(rootdir, f))
    return file_list

def get_stock_from_path(pathname):
    return os.path.splitext(os.path.split(pathname)[-1])[0]

def _judge(df, window):
    df["close_shift"] = df["close"].shift(-1 * window)
    df["label" + str(window)] = df["close_shift"]/df["close"]
    del df["close_shift"]
    return df

def judge(df):
    for i in range(1, 52):
        df = _judge(df, i)
    return df

def filter(df):
    return False

    mean =  np.max(np.abs((df.tail(11)["label1"].head(10).values - 1)))
    if mean < 0.01:
        return True
    if np.max(df.tail(11)["volume"].head(10).values) < 500000:
        return True
    return False

def get_pd(symbol):
    names = ["date", 'open', 'high', 'low', 'close', 'volume', 'adjclose']
    df = pd.read_csv(os.path.join(local_path, '..', 'data', 'yeod', symbol+".csv"), \
            header = None, names = names, \
            dtype = {"volume":np.float64}, \
            skiprows=1, index_col = 'date', parse_dates=True).sort_index()
    return df


def get_eod(symbol):
    df = pd.read_csv(symbol)
    if df["volume"].mean() < 10000:
        return None
    if df["close"].mean() < 10:
        return None
    return df[df["volume"]>0]

def _one_work(eod, func, dir_out):
    symbol = get_stock_from_path(eod)
    df = get_eod(eod)
    if df is None:
        return
    df = getattr(ta, func)(df)
    df = judge(df)
    if not os.path.isdir(dir_out):
        os.mkdir(dir_out)
    if not filter(df):
        df.to_pickle(os.path.join(dir_out, symbol+".pkl"))
    else:
        print symbol, False

def work(pool_num, dir_data, func, dir_out):
    if not os.path.exists(dir_out):
        os.makedirs(dir_out)
    pool = multiprocessing.Pool(processes=pool_num)
    result = []
    for each in get_file_list(dir_data):
        result.append(pool.apply_async(_one_work, (each, func, dir_out)))
        #_one_work(each, func, dir_out)
    pool.close()
    pool.join()
    #for each in result:
    #    print each.get()

def main(argv):
    eod = argv[0]
    ta = argv[1]
    pool_num = int(argv[2])
    work(pool_num,
         os.path.join(root, 'data', 'yeod', eod),
         ta,
         os.path.join(root, 'data', 'ta', ta+"_"+eod)
    )
if __name__ == '__main__':
    main(sys.argv[1:])

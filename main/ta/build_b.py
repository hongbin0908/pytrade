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

#{{{ judge
def _judge(df, window):
    df["close_shift"] = df["close"].shift(-1 * window)
    df["label" + str(window)] = df["close_shift"]/df["close"]
    del df["close_shift"]
    return df

def judge(df):
    for i in range(1, 6):
        df = _judge(df, i)
    return df
# }}}

#{{{ judge2
def judge2(df):
    for i in range(1, 6):
        df["labelii"+str(i)] = df["label"+str(i)] / df["ta_index_hdiff_close_%d" % i].shift(-1*i)
    return df
# }}}

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
    print symbol
    df = get_eod(eod)
    if df is None:
        print symbol
        return
    df = getattr(ta, func)(df)
    df = judge(df)
    if 'ta_index_hdiff_close_1' in df.columns:
        df = judge2(df)
    if not os.path.isdir(dir_out):
        os.mkdir(dir_out)
    df.to_pickle(os.path.join(dir_out, symbol+".pkl"))

def work(pool_num, dir_data, func, dir_out):
    if not os.path.exists(dir_out):
        os.makedirs(dir_out)
    pool = multiprocessing.Pool(processes=pool_num)
    result = []
    for each in get_file_list(dir_data):
        if pool_num <= 1:
            _one_work(each, func, dir_out)
        else:
            result.append(pool.apply_async(_one_work, (each, func, dir_out)))
    pool.close()
    pool.join()
    #for each in result:
    #    print each.get()

def main(argv):
    eod = argv[0]
    batch = int(argv[1])
    ta = argv[2]
    pool_num = int(argv[3])
    eod_father = os.path.join(root, 'data', 'yeod_batch', eod + "-" + str(batch))
    for  d in os.listdir(eod_father):
        if d == None or not os.path.isdir(os.path.join(eod_father,d)):
            continue
        
        work(pool_num,
             os.path.join(eod_father, d),
             ta,
             os.path.join(root, 'data', 'ta_batch', ta+"-"+eod+"-"+str(batch),str(d))
    )
if __name__ == '__main__':
    main(sys.argv[1:])

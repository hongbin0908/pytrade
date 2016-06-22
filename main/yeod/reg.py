#!/usr/bin/env python2.7

import os, sys
import finsymbols
import pandas as pd
import numpy as np
import multiprocessing

local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..')
sys.path.append(os.path.join(root,'..'))

from main.ta import build


def get_eod_path(eod):
    return os.path.join(root, 'data', 'eod', eod)

def get_reg_path(eod):
    p = os.path.join(root, 'data', 'reg', eod)
    if not os.path.exists(p):
        os.makedirs(p)
    return p


def get_eod(symbol):
    names = ["date", 'open', 'high', 'low', 'close', 'volume', 'adjclose']
    df = pd.read_csv(symbol, \
            header = None, names = names, \
            dtype = {"volume":np.float64}, \
            skiprows=1, parse_dates=True)

    if df["volume"].mean() < 10000:
        return None
    if df["close"].mean() < 10:
        return None

    df = df.sort_values(["date"], ascending=True)
    return df[df["volume"]>0]

def _one_work(from_, to_, sym):
    sympath = os.path.join(from_, sym + ".csv")
    df = get_eod(sympath)
    first = df["adjclose"].values[0]
    df["adjclose"] = df["adjclose"]/first
    df["open"] = df["open"] * df["adjclose"]/df["close"]
    df["high"] = df["high"] * df["adjclose"]/df["close"]
    df["low"]  = df["low"]   * df["adjclose"]/df["close"]
    df["close"]= df["close"] * df["adjclose"]/df["close"]
    df["volume"] = df["volume"] * df["adjclose"]/df["close"]
    outfile = os.path.join(to_,sym + ".csv")
    print outfile
    df.to_csv(outfile)

def get_stock_from_path(pathname):
    return os.path.splitext(os.path.split(pathname)[-1])[0]

def work(pool_num, eod):
    pool = multiprocessing.Pool(processes=pool_num)
    result = []
    for each in build.get_file_list(get_eod_path(eod)):
        #result.append(pool.apply_async(_one_work, (each,)))
        sym = get_stock_from_path(each)
        _one_work(get_eod_path(eod), get_reg_path(eod), sym)
    pool.close()
    pool.join()
    for each in result:
        print each.get()

def main(argv):
    eod = argv[1]
    pool_num = int(argv[2])
    work(pool_num, eod)
if __name__ == '__main__':
    main(sys.argv)

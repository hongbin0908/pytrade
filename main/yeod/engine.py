#!/usr/bin/env python3

import os, sys, time
import pandas as pd
import numpy as np
import multiprocessing
import logging

local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..', '..')
sys.path.append(root)

from main.base import stock_fetcher as sf

logging.basicConfig(level=logging.WARN)
local_path = os.path.dirname(__file__)

def assert_valid(df):
    assert len(df.replace([np.inf,-np.inf],np.nan).dropna()) == len(df)

def _single(symbol, data_dir):
    df = sf.get_stock(symbol)
    if df is None:
        df = sf.get_stock3(symbol)
    if df is None:
        return 0
    assert_valid(df)
    df = df.reset_index()
    df[df.volume > 1].round(6).to_csv(os.path.join(data_dir, symbol + ".csv"), index=False, date_format='%Y-%m-%d')
    return len(df)
def work(syms,data_dir, processes):
    logging.debug("data_dir : %s" % data_dir)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    syms.sort()
    pool = multiprocessing.Pool(processes = int(processes) )
    result = {}
    for sym in syms:
        if sym.find('^') > 0:
            continue
        if sym.find('.') > 0:
            continue
        #if os.path.isfile(os.path.join(data_dir, sym + ".csv")):
        #    continue
        if processes <= 1:
            _single(sym, data_dir)
        else:
            pool.apply_async(_single, (sym, data_dir))
    pool.close()
    pool.join()
    succ = 0; fail = 0
    for each in result:
        if result[each] > 0: succ += 1
        else: assert(False)

    return fail

if __name__ == "__main__":
    df1 = sf.get_stock("YHOO")
    df2 = sf.get_stock2("YHOO")
    pd.set_option('display.expand_frame_repr', False) 
    print(df1.tail())
    print(df2.tail())

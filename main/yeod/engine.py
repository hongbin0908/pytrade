#!/usr/bin/env python3

import os, sys, time
import pandas as pd
import numpy as np
import pandas_datareader.yahoo.daily as yahoo
import multiprocessing
import logging

logging.basicConfig(level=logging.DEBUG)

local_path = os.path.dirname(__file__)

def get_stock(symbol):
    import urllib.request
    count = 3
    while count >= 0 :
        response = urllib.request.urlopen('https://www.quandl.com/api/v1/datasets/WIKI/%s.csv?auth_token=GcGVd3Q685QszTrWDhud' % symbol)
        try:
            df = pd.read_csv(response)
        except Exception as exc:
            print('%r generated an exception: %s' % (symbol, exc))
            count -= 1
            continue
        if (len(df) > 10):
            print(symbol, "len < 10")
            time.sleep(10)
            count -= 1
            continue
        break
    names = ['date', 'openo', 'higho', 'lowo', 'closeo', 'volumeo', 'dividend', 'ratio', 'open', 'high', 'low', 'close', 'volume']
    df.columns = names
    df = df.dropna()
    df.sort_values("date", ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df[["date", "open", "high", "low", "close", "volume"]]


def get_stock2(symbol):
    """
    deprecaded use get_stock(quandl version) instead
    """
    yeod = yahoo.YahooDailyReader(symbol, "17000101", "20990101", adjust_price=True)
    df = yeod.read()
    df.reset_index(drop = False, inplace=True)
    names = ['date', 'open', 'high', 'low', 'close', 'volume', 'adjrate']
    df.columns = names
    df= df.dropna()
    return df


def assert_valid(df):
    assert len(df.replace([np.inf,-np.inf],np.nan).dropna()) == len(df)

def _single(symbol, data_dir):
    df = get_stock2(symbol)
    assert_valid(df)
    df.round(6).to_csv(os.path.join(data_dir, symbol + ".csv"), index=False, date_format='%Y-%m-%d')

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
    get_stock("GOOG")

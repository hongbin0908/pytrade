#!/usr/bin/env python3

import os, sys, time
import pandas as pd
import numpy as np
import pandas_datareader.yahoo.daily as yahoo
import multiprocessing
import logging

logging.basicConfig(level=logging.WARN)

local_path = os.path.dirname(__file__)

def get_stock(symbol):
    import urllib.request
    count = 1
    while count > 0 :
        url = 'https://www.quandl.com/api/v3/datatables/WIKI/PRICES.csv?ticker=%s&api_key=jmNW9q_f2LYzA9fszZ33' % symbol

        print(url)
        response = urllib.request.urlopen(url)
        try:
            df = pd.read_csv(response)
            print(len(df))
        except Exception as exc:
            print('%r generated an exception: %s' % (symbol, exc))
            count -= 1
            continue
        if (len(df) < 10):
            print(symbol, "len < 10")
            time.sleep(10)
            count -= 1
            continue
        break
    print(df.columns)
    names = ['sym', 'date', 'openo', 'higho', 'lowo', 'closeo', 'volumeo', 'dividend', 'ratio', 'open', 'high', 'low', 'close', 'volume']
    df.columns = names
    df = df.dropna()
    df.sort_values("date", ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df = df.set_index("date")
    df = df[['openo', 'higho', 'lowo', 'closeo', 'open', 'high', 'low', 'close', 'volume']]
    return df.round(6)


def get_stock2(symbol):
    """
    deprecaded use get_stock(quandl version) instead
    """
    try:
        yeod = yahoo.YahooDailyReader(symbol, "17000101", "20990101", adjust_price=False)
        df = yeod.read()
    except:
        return None
    df.reset_index(drop = False, inplace=True)
    names = ['date', 'openo', 'higho', 'lowo', 'closeo', 'volumeo', 'adjclose']
    df.columns = names
    df= df.dropna()
    df = df.set_index("date")
    yeod = yahoo.YahooDailyReader(symbol, "17000101", "20990101", adjust_price=True)
    df2 = yeod.read()
    if len(df2) < 10:
        assert(False)
        return None
    df2.reset_index(drop = False, inplace=True)
    names = ['date', 'open', 'high', 'low', 'close', 'volume', 'ratio']
    df2.columns = names
    df2= df2.dropna()
    df2 = df2.set_index("date")

    df = pd.concat([df,df2], axis=1, join_axes=[df.index])
    assert len(df.shape) == len(df2.shape)
    df = df[['openo', 'higho', 'lowo', 'closeo', 'open', 'high', 'low', 'close', 'volume']]
    return df.round(6)

def get_stock3(symbol):
    import urllib.request
    count = 1
    while count > 0 :
        url = 'http://hongindex.com/yeod/dead_20170304/%s.csv' % symbol
        print(url)
        response = urllib.request.urlopen(url)
        try:
            df = pd.read_csv(response)
            print(len(df))
        except Exception as exc:
            print('%r generated an exception: %s' % (symbol, exc))
            count -= 1
            continue
        if (len(df) < 10):
            print(symbol, "len < 10")
            time.sleep(10)
            count -= 1
            continue
        break
    df = df.dropna()
    df.sort_values("date", ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df = df.set_index("date")
    df = df[['openo', 'higho', 'lowo', 'closeo', 'open', 'high', 'low', 'close', 'volume']]
    return df.round(6)

def assert_valid(df):
    assert len(df.replace([np.inf,-np.inf],np.nan).dropna()) == len(df)

def _single(symbol, data_dir):
    df = get_stock2(symbol)
    if df is None:
        df = get_stock3(symbol)
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
    df1 = get_stock("YHOO")
    df2 = get_stock2("YHOO")
    pd.set_option('display.expand_frame_repr', False) 
    print(df1.tail())
    print(df2.tail())

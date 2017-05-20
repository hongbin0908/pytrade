#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#@author  Bin Hong

"""
the base method use by pytrade
"""

import sys,os
import urllib.request
import time
import pandas as pd
import pandas_datareader.yahoo.daily as yahoo
import json

local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..', '..')
sys.path.append(root)

def get_stock_once(symbol):
    is_first = True
    while True:
        if is_first:
            url = 'https://www.quandl.com/api/v3/datatables/WIKI/PRICES.json?ticker=%s&api_key=77sr5UvZ2qs5z38i_Hf5' % symbol
        elif not response['meta']['next_cursor_id'] is None:
            url = 'https://www.quandl.com/api/v3/datatables/WIKI/PRICES.json?ticker=%s&qopts.cursor_id=%s&api_key=77sr5UvZ2qs5z38i_Hf5' % (symbol, response['meta']['next_cursor_id'])
        else:
            return df
        print(url)
        response = urllib.request.urlopen(url)
        response = response.read().decode('utf8')
        response = json.loads(response)
        if is_first:
            df = pd.DataFrame(response['datatable']['data'])
        else:
            df = pd.concat([df, pd.DataFrame(response['datatable']['data'])])
        if is_first:
            is_first = False
def get_stock(symbol):
    count = 1
    while count > 0 :
        try:
            df = get_stock_once(symbol)
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
        url = 'http://www.hongindex.com/yeod/dead_20170304/%s.csv' % symbol
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

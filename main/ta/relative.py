#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

#@author Bin Hong
import os,sys
import concurrent.futures
import platform
import traceback

import pandas as pd
import multiprocessing

local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..', '..')
sys.path.append(root)

import main.base as base
from main.base.score2 import ScoreLabel
from main.base.timer import Timer
from main.ta import ta_set
import talib
from datetime import datetime
from main.model import bitlize
from main.model import feat_select


def is_trend_long(df):
    ma = talib.MA(df.close.values, timeperiod=10)
    i = 1000
    if i > len(ma) - 1:
        i = len(ma) - 1
    if ma[i] / ma[15] < 2.0:
        return True
    return False


def _one_work(df_index, sym, confer, dirname = ""):
    try:
        if not os.path.exists(os.path.join(base.dir_eod(), dirname, sym + ".csv")):
            print("Not exsits %s!!!!!!" % os.path.join(base.dir_eod(), dirname, sym + ".csv"))
            return None
        df = pd.read_csv(os.path.join(base.dir_eod(),dirname, sym + ".csv"))
        #df = df[["date", "open", "high", "low", "close", "volume"]]
        df[['volume']] = df[["volume"]].astype(float)
        if df is None:
            print(sym)
            return
        df["sym"] = sym
        origin_len = len(df)
        if (len(df) < 300):
            print(sym, "too short!")
            return None
        if (len(df[df.close/df["adjrate"]< 10])/len(df) > 0.5):
            print(sym, "price too low!")
            return None
        if (len(df[df.volume< 100000])/len(df) > 0.5):
            print(sym, "volume too low!")
            return None
        if len(df[df["high"] < df['close']])>0 or len(df[df["low"] > df["close"]])>0:
            print(sym, "high < close or low > close ")
            return None

        df_index.set_index("date", drop=True, inplace=True)
        df.set_index("date", drop=True, inplace=True)
        df_index = df_index[["close"]].rename(columns={"close":"iclose"})
        df = pd.concat([df, df_index], axis=1, join='inner')
        #df["close"] = df["close"]/df["iclose"]*2000
        #df["open"]  = df["open"]/df["iclose"]*2000
        #df["high"]  = df["high"]/df["iclose"]*2000
        #df["low"]  = df["low"]/df["iclose"]*2000

        df.reset_index(drop=False).to_csv(os.path.join(base.dir_eod(), dirname, sym+".rel.csv"))
        if len(df) < 100:
            return None
        return df
    except:
        traceback.print_exc()
        assert False


def work(pool_num, symset, confer, dirname = ""):
    to_apends = []
    Executor = concurrent.futures.ProcessPoolExecutor
    df_index = pd.read_csv(os.path.join(base.dir_eod(), "index", "^GSPC.csv"))
    with Executor(max_workers=pool_num) as executor:
        futures = {executor.submit(_one_work, df_index, sym, confer, dirname): sym for sym in symset}
        for future in concurrent.futures.as_completed(futures):
            sym = futures[future]
            try:
                data = future.result()
                if data is None:
                    continue
            except Exception as exc:
                traceback.print_exc()
                executor.shutdown(wait=False)
                sys.exit(1)

if __name__ == '__main__':
    ta = ta_set.TaSetBase1Ext8()
    df = _one_work("AAPL", ta)
    print(is_trend_long(df))

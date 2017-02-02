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


def is_trend_long(df):
    ma = talib.MA(df.close.values, timeperiod=10)
    i = 1000
    if i > len(ma) - 1:
        i = len(ma) - 1
    if ma[i] / ma[15] < 2.0:
        return True
    return False

def _one_work(sym, ta, confer, dirname = ""):
    try:
        if not os.path.exists(os.path.join(base.dir_eod(), dirname, sym + ".csv")):
            print("Not exsits %s!!!!!!" % os.path.join(base.dir_eod(), dirname, sym + ".csv"))
            return None
        df = pd.read_csv(os.path.join(base.dir_eod(),dirname, sym + ".csv"))
        df = df[["date", "open", "high", "low", "close", "volume"]]
        df[['volume']] = df[["volume"]].astype(float)
        if df is None:
            print(sym)
            return
        df["sym"] = sym
        df = ta.get_ta(df, confer)
        df.to_pickle(os.path.join(base.dir_ta(ta.get_name()), sym+".pkl"))
        return df
    except:
        traceback.print_exc()
        assert False


def bit_apply(df, name, fname, start, end):
    try:
        print(name, fname, start, end)
        new = df.apply(lambda row: 1 if ((row[fname] >= start) and (row[fname] < end)) else 0, axis=1)
        s = pd.Series(new, name=name)
        return s
    except:
        traceback.print_exc()
        assert False

def work(pool_num, symset, ta, scores, confer, dirname = ""):
    print(pool_num)
    to_apends = []
    Executor = concurrent.futures.ProcessPoolExecutor
    with Executor(max_workers=pool_num) as executor:
        futures = {executor.submit(_one_work, sym, ta, confer, dirname): sym for sym in symset}
        for future in concurrent.futures.as_completed(futures):
            sym = futures[future]
            try:
                data = future.result()
                if data is None:
                    continue
                if (len(data) < 300):
                    print(sym, "too short!")
                    continue
                for score in scores:
                    assert isinstance(score, ScoreLabel)
                    data = score.agn_score(data)
                    data = data[data.ta_NATR_7 > 1.0]
                to_apends.append(data)
                print(sym)
            except Exception as exc:
                traceback.print_exc()
                executor.shutdown(wait=False)
                sys.exit(1)
    #longs = []
    #shorts = []
    #for each in to_apends:
    #    if is_trend_long(each):
    #        longs.append(each)
    #    else:
    #        shorts.append(each)
    #print("longs: ", len(longs))
    #print("shorts: ", len(shorts))
    #return pd.concat(longs)
    df = pd.concat(to_apends)
    from main.model import bitlize
    with Timer("bitlize.feat_select") as t:
        df_bit1 = bitlize.feat_select(df, 0.8, confer.score1.get_name(), 1, 100, confer.n_pool)
        #df_bit2 = bitlize.feat_select(df, 0.8, confer.score1.get_name(), 2, 100, confer.n_pool)
        #df_bit = pd.concat([df_bit1, df_bit2], axis = 0)
        #assert len(df_bit1) + len(df_bit2) == len(df_bit)
        df_bit = df_bit1

    tobe = [df[["sym", "date", "open", "high", "low", "close","volume", confer.score1.get_name()]]]


    Executor = concurrent.futures.ProcessPoolExecutor
    with Timer("bit_apply") as t:
        with Executor(max_workers=confer.n_pool) as executor:
            futures = {
                executor.submit(bit_apply, df[[each["fname"]]].copy(), each["name"], each["fname"], each["start"],
                                each["end"]): (i, each) for (i, each) in df_bit.iterrows()}
            for future in concurrent.futures.as_completed(futures):
                try:
                    i, each = futures[future]
                    tobe.append(future.result())
                except:
                    traceback.print_exc()
                    executor.shutdown(wait=False)
                    sys.exit(1)

    result = pd.concat(tobe, axis=1)
    assert len(df) == len(result)
    ## 防止一致正在下跌的股票会持续被选中, 因此只对周一到周五的股票进行预测. 
    if conf.score1.get_name.startswith("score_label_5"):
        print("filter....")
        result = result[result.apply(lambda x: datetime.strptime(x['date'], "%Y-%m-%d").weekday()==1)]

    return result


if __name__ == '__main__':
    ta = ta_set.TaSetBase1Ext8()
    df = _one_work("AAPL", ta)
    print(is_trend_long(df))

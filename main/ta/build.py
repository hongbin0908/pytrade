#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

#@author Bin Hong
import os,sys
import pandas as pd
import multiprocessing
local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..', '..')
sys.path.append(root)

import main.base as base
import main.ta as ta
import main.yeod.yeod as yeod


#{{{ judge
def _judge(df, window):
    df["close_shift"] = df["close"].shift(-1 * window)
    df["label" + str(window)] = df["close_shift"]/df["close"]
    del df["close_shift"]
    return df


def judge(df):
    for i in range(1, 30):
        df = _judge(df, i)
    return df
# }}}


#{{{ judge2
def judge2(df):
    for i in range(1, 30):
        df["labelii"+str(i)] = df["label"+str(i)] / df["ta_index_hdiff_close_%d" % i].shift(-1*i)
    return df
# }}}


def _one_work(sym, taname):
    impstr = "import ta_%s as conf" % taname
    exec impstr
    df = pd.read_csv(os.path.join(base.dir_eod(),sym + ".csv"))
    if df is None:
        print sym
        return
    df = judge(df)
    if 'ta_index_hdiff_close_1' in df.columns:
        df = judge2(df)
    df["sym"] = sym
    df = conf.main(df)
    df.to_pickle(os.path.join(base.dir_ta(taname), sym+".pkl"))


def _one_work2(sym, tafunc):
    exec "import main.pandas_talib.sig_%s as conf" % tafunc
    df = pd.read_csv(os.path.join(base.dir_eod(),sym + ".csv"))
    if df is None:
        print sym
        return
    df = judge(df)
    if 'ta_index_hdiff_close_1' in df.columns:
        df = judge2(df)
    df["sym"] = sym
    df = conf.main(df)
    df.to_pickle(os.path.join(base.dir_ta(conf.__name__), sym+".pkl"))


def work(pool_num, setname, taname):
    pool = multiprocessing.Pool(processes=pool_num)
    result = []
    for sym in getattr(yeod, "get_%s" % setname)():
        if pool_num <= 1:
            _one_work(sym, taname)
        else:
            result.append(pool.apply_async(_one_work, (sym, taname)))
    pool.close()
    pool.join()
    for each in result:
        re = str(each.get())
        if re != "None":
            print each.get()


def work2(pool_num, setname, taname):
    pool = multiprocessing.Pool(processes=pool_num)
    result = []
    for sym in getattr(yeod, "get_%s" % setname)():
        if pool_num <= 1:
            _one_work2(sym, taname)
        else:
            result.append(pool.apply_async(_one_work2, (sym, taname)))
    pool.close()
    pool.join()
    for each in result:
        re = str(each.get())
        if re != "None":
            print each.get()


def main(args):
    work(args.poolnum,
             args.setname,
             args.taname)
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="build technical analysis")
    parser.add_argument('-p', '--pool', help="thread pool num", dest="poolnum", action="store", default=1, type=int)
    parser.add_argument('setname', help = "the sym set to be ta")
    parser.add_argument('taname', help = "the sym set to be ta")
    args = parser.parse_args()
    main(args)

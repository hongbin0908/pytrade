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
from main.score.score import ScoreLabel

def _one_work(sym, confer, dirname = ""):
    filename = os.path.join(base.dir_eod(), dirname, sym + ".csv")
    try:
        if not os.path.exists(filename):
            print("Not exsits %s!!!!!!" % filename)
            return None
        df = pd.read_csv(filename)
        #df = df[["date", "open", "high", "low", "close", "volume"]]
        df[['volume']] = df[["volume"]].astype(float)
        if df is None:
            print(sym)
            return
        df["sym"] = sym
        for score in confer.scores:
            df = score.agn_score(df)
        return df
    except:
        traceback.print_exc()
        assert False


def work(pool_num, symset, confer, dirname = ""):
    if not os.path.exists(confer.get_score_file()) or  confer.force:
        to_apends = []
        Executor = concurrent.futures.ProcessPoolExecutor
        with Executor(max_workers=pool_num) as executor:
            futures = {executor.submit(_one_work, sym, confer, dirname): sym for sym in symset}
            for future in concurrent.futures.as_completed(futures):
                sym = futures[future]
                try:
                    data = future.result()
                    if data is None:
                        continue
                    to_apends.append(data)
                except Exception as exc:
                    traceback.print_exc()
                    executor.shutdown(wait=False)
                    sys.exit(1)
        df = pd.concat(to_apends)
        df = df.sort_values(["sym", "date"])
        df.reset_index(drop=True).to_pickle(confer.get_score_file())

def work_with_original_fea(pool_num, symset, confer, dirname = ""):
    if not os.path.exists(confer.get_score_with_original_file()) or  confer.force:
        to_apends = []
        Executor = concurrent.futures.ProcessPoolExecutor
        with Executor(max_workers=pool_num) as executor:
            futures = {executor.submit(_one_work, sym, confer, dirname): sym for sym in symset}
            for future in concurrent.futures.as_completed(futures):
                sym = futures[future]
                try:
                    data = future.result()
                    if data is None:
                        continue
                    to_apends.append(data)
                except Exception as exc:
                    traceback.print_exc()
                    executor.shutdown(wait=False)
                    sys.exit(1)
        df = pd.concat(to_apends)
        df = df.sort_values(["sym", "date"])

        df_ta = pd.read_pickle(confer.get_ta_file())
        df_ta["sd"] = df_ta["sym"] + df_ta["date"]
        df_ta = df_ta.set_index("sd")

        df["sd"] = df["sym"] + df["date"]
        df = df.set_index("sd")

        df_merge = pd.concat([df_ta, df[[score.get_name() for score in confer.scores]]], axis=1,
                       join_axes=[df_ta.index])

        df_merge.reset_index(drop=True).to_pickle(confer.get_score_with_original_file())
"""
def work_with_original_fea(pool_num, symset, confer, dirname = ""):
    if not os.path.exists(confer.get_score_with_original_file()) or  confer.force:
        to_apends = []
        for sym in symset:
            data = _one_work(sym, confer, dirname)
            to_apends.append(data)

        df = pd.concat(to_apends)
        df = df.sort_values(["sym", "date"])

        df_ta = pd.read_pickle(confer.get_ta_file())
        df_ta["sd"] = df_ta["sym"] + df_ta["date"]
        df_ta = df_ta.set_index("sd")

        df["sd"] = df["sym"] + df["date"]
        df = df.set_index("sd")

        df_merge = pd.concat([df_ta, df[[score.get_name() for score in confer.scores]]], axis=1,
                       join_axes=[df_ta.index])

        df_merge.reset_index(drop=True).to_pickle(confer.get_score_with_original_file())
"""

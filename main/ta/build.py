#!/usr/bin/env python
# -*- coding: utf-8 -*-

#@author 
import os,sys
import talib
import pandas as pd
import numpy as np
local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..')
sys.path.append(root)
import ta


def get_file_list(rootdir):    
    file_list = []
    for f in os.listdir(rootdir):   
        if f == None or not f.endswith(".csv"): 
            continue
        file_list.append(os.path.join(rootdir, f))
         
    return file_list


def get_stock_from_path(pathname):
    return os.path.splitext(os.path.split(pathname)[-1])[0]

def _judge(df, window):
    df["close_shift"] = df["close"].shift(-1 * window)
    df["label" + str(window)] = df["close_shift"]/df["close"]
    del df["close_shift"]
    return df 

def judge(df):
    df = _judge(df, 1)
    df = _judge(df, 2)
    df = _judge(df, 3)
    df = _judge(df, 4)
    df = _judge(df, 5)
    df = _judge(df, 6)
    df = _judge(df, 8)
    df = _judge(df, 10)
    df = _judge(df, 15)
    df = _judge(df, 20)
    df = _judge(df, 30)
    df = _judge(df, 60)
    return df
def get_pd(symbol):
    names = ["date", 'open', 'high', 'low', 'close', 'volume', 'adjclose']
    df = pd.read_csv(os.path.join(local_path, '..', 'data', 'yeod', symbol+".csv"), \
            header = None, names = names, \
            dtype = {"volume":np.float64}, \
            skiprows=1, index_col = 'date', parse_dates=True).sort_index()
    return df

def get_eod(symbol):
    names = ["date", 'open', 'high', 'low', 'close', 'volume', 'adjclose']
    df = pd.read_csv(symbol, \
            header = None, names = names, \
            dtype = {"volume":np.float64}, \
            skiprows=1, index_col = 'date', parse_dates=True).sort_index()
    
    if df["volume"].mean() < 10000:
        return None
    if df["close"].mean() < 10:
        return None

    return df[df["volume"]>0]
def main1():
    for each in get_file_list(os.path.join(local_path, '..', 'data', 'yeod')):
        symbol = get_stock_from_path(each)
        df = get_eod(each)
        if df is None:
            continue
        df = ta.cal_all(df)
        df = judge(df)
        dir_out = os.path.join(root, 'data', 'ta1')
        if not os.path.isdir(dir_out):
            os.mkdir(dir_out)
        df.to_csv(os.path.join(dir_out, symbol + ".csv"))
def main2():
    for each in get_file_list(os.path.join(local_path, '..', 'data', 'yeod')):
        symbol = get_stock_from_path(each)
        df = get_eod(each)
        if df is None:
            continue
        df = ta.call2(df)
        df = judge(df)
        dir_out = os.path.join(root, 'data', 'ta2')
        if not os.path.isdir(dir_out):
            os.mkdir(dir_out)
        df.to_csv(os.path.join(dir_out, symbol + ".csv"))

def main3():
    l = get_file_list(os.path.join(local_path, '..', 'data', 'yeod_full')); l.sort()
    #l = l[:5]
    for each in l:
        symbol = get_stock_from_path(each)
        df = get_eod(each)
        if df is None:
            continue
        print symbol, df.shape
        df = ta.cal_all(df)
        df = judge(df)

        dir_out = os.path.join(root, 'data', 'ta3')
        if not os.path.isdir(dir_out):
            os.mkdir(dir_out)
        df.to_csv(os.path.join(dir_out, symbol + ".csv"))
def main4():
    for each in get_file_list(os.path.join(local_path, '..', 'data', 'yeod_full')):
        symbol = get_stock_from_path(each)
        df = get_eod(each)
        if df is None:
            continue
        df = ta.cal_all(df)
	df = df[[
		 'open',
		 'high',
		 'low',
		 'close',
		 'volume',
		 'adjclose',
		 'ta_diff_close_0_1',
                 'ta_atr_14',
                 'ta_diff_close_1_1', 
                 'ta_trange', 
                 'ta_mdi_14',
                 'ta_natr_14',
                 'ta_diff_ta_mdi_14_0_1', 
                 'ta_aroon_up_14', 
                 'ta_adsoc', 
                 'ta_diff_close_2_1', 
                 'ta_diff_close_11_1', 
                 'ta_diff_ta_pdi_14_0_1', 
                 'ta_diff_close_3_1', 
                 'ta_diff_close_5_1', 
                 'ta_diff_close_6_1', 
                 'ta_diff_close_4_1', 
                 'ta_pdi_14', 
                 'ta_diff_ta_pdi_14_4_1', 
                 'ta_diff_close_9_1', 
                 'ta_diff_ta_pdi_14_3_1', 
                 'ta_diff_close_10_1', 
                 'ta_adxr_14', 
                 'ta_aroon_down_14', 
                 'ta_obv', 
                 'ta_ad', 
                 'ta_apo_12_26_0', 
                 'ta_diff_ta_pdi_14_2_1', 
                 'ta_diff_close_12_1', 
                 'ta_diff_ta_pdi_14_1_1', 
                 'ta_aroonosc_14',
]]
        df = judge(df)
        dir_out = os.path.join(root, 'data', 'ta4')
        if not os.path.isdir(dir_out):
            os.mkdir(dir_out)
        df.to_csv(os.path.join(dir_out, symbol + ".csv"))
if __name__ == '__main__':
    #main3()
    main1()
    #main2()
    #main4()

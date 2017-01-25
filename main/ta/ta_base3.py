#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author hongbin@youzan.com

import os,sys
import talib
import numpy as np
import pandas as pd

local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..', '..')
sys.path.append(root)

from main.utils import time_me
import main.pandas_talib as pta
import main.base as base


def main(df):
    df = df.reset_index(drop=True)
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes=df['close'].values
    volumes = df['volume'].values

    #Cycle Indicators': {{{
    df['ta_HT_DCPERIOD'] = talib.HT_DCPERIOD(closes)
    df['ta_HT_DCPHASE'] =  talib.HT_DCPHASE (closes)
    #print len(talib.HT_PHASOR(closes)), len(closes)
    #print talib.HT_PHASOR(closes) 

    #print 'ssssss'

    #print closes
    #df['ta_HT_PHASOR'] = talib.HT_PHASOR(closes)/closes
    #df['ta_HT_SINE'] =   talib.HT_SINE(closes)/closes
    #df['ta_HT_TRENDMODE'] = talib.HT_TRENDLINE(closes)/closes
    #}}}

    #'Momentum Indicators': {{{
    for i in [7,10,14,28]:
        df['ta_ADX_%d' % i] = talib.ADX(highs, lows, closes, i)
        df['ta_ADXR_%d' % i] = talib.ADXR(highs, lows, closes, i)
    for couple in [(6, 13), (12, 26)]:
        df['ta_APO_%d_%d' % (couple[0], couple[1])] = talib.APO(closes, couple[0], couple[1]) / closes
    for i in range(7, 28):
        #df['ta_AROON_%d' % i] = talib.AROON(highs, lows, i)/closes
        df['ta_AROONOSC_%d' % i] = talib.AROONOSC(highs, lows, i)/closes
    df['ta_BOP'] = talib.BOP(opens, highs, lows, closes)
    df['ta_CCI'] = talib.CCI(highs, lows, closes)
    for i in [7,10,14,28]:
        df['ta_CMO_%d' % i] = talib.CMO(closes, i)
        #df['ta_DX_%d' %i] = talib.DX(closes, i)/closes
    for tri in [(6,13,8),(12, 26, 9)]:
        macd = talib.MACD(closes, tri[0], tri[1], tri[2]) / closes
        df['ta_MACD_macd_%d_%d_%d'%(tri[0],tri[1],tri[2])] = macd[0]
        df['ta_MACD_macdsignal_%d_%d_%d'%(tri[0],tri[1],tri[2])] = macd[1]
        df['ta_MACD_macdhist_%d_%d_%d'%(tri[0],tri[1],tri[2])] = macd[2]
    
    #for i in range(6,18):
    #    macd = talib.MACDFIX(closes,i)
    #    df['ta_MACDFIX_macd_%d'% i]= macd[0]
    #    df['ta_MACDFIX_macdsignal_%d'% i]= macd[1]
    #    df['ta_MACDFIX_macdhist_%d'% i]= macd[2]
    for i in [7,10,14,28]:
        df['ta_MFI_%d'%i] = talib.MFI(highs,lows,closes,volumes)
    for i in [7,10,14,28]:
        df['ta_MINUS_DI_%d'%i]  = talib.MINUS_DI(highs,lows,closes,i)
        df['ta_MINUS_DM_%d'%i]  = talib.MINUS_DI(highs,lows,closes,i)
        df['ta_MOM_%d' % i]     = talib.MOM(closes, i)/closes
        df['ta_PLUS_DI_%d' % i] = talib.PLUS_DI(highs,lows,closes,i)
        df['ta_PLUS_DM_%d' % i] = talib.PLUS_DM(highs,lows,i)
    for couple in [(6,13),(12,26)]:
        df['ta_PPO_%d_%d'%(couple[0],couple[1])] = talib.PPO(closes, couple[0],couple[1])
    for i in [2,5,7,10,14,28]:
        df['ta_ROC_%d'% i] = talib.ROC(closes, i)
        df['ta_ROCP_%d'%i] = talib.ROCP(closes, i)
        df['ta_ROCR_%d'%i] = talib.ROCR(closes, i)
        df['ta_ROCR100_%d'%i] = talib.ROCR100(closes,i)
        df['ta_RSI_%d'%i] = talib.RSI(closes, i)
    for tri in [(5,3,3),(10,6,6),(20,12,12)]:
        stoch = talib.STOCH(highs,lows,closes, fastk_period=tri[0],slowk_period=tri[1],slowd_period=tri[2])
        df['ta_STOCH_slowk_%d_%d_%d' % (tri[0],tri[1],tri[2])] = stoch[0]
        df['ta_STOCH_slowd_%d_%d_%d' % (tri[0],tri[1],tri[2])] = stoch[1]
    for i in [2,5,7,10,14,28]:
        for c in [(5,3),(10,6),(20,12)]:
            stochrsi = talib.STOCHRSI(closes,i,c[0],c[1])
            df['ta_STOCHRSI_slowk_%d_%d_%d'%(i,c[0],c[1])] = stochrsi[0]
            df['ta_STOCHRSI_slowd_%d_%d_%d'%(i,c[0],c[1])] = stochrsi[1]
    for i in [2,5,7,10,14,28]:
        df['ta_TRIX_%d'%i] = talib.TRIX(closes,i)
    for tr in [(7,14,28)]:
        df['ta_ULTOSC_%d_%d_%d'%(tr[0],tr[1],tr[2])] = talib.ULTOSC(highs,lows, closes, tr[0],tr[1],tr[2])
    for i in [2,5,7,10,14,28]:
        df['ta_WILLR_%d'%i] = talib.WILLR(highs, lows,closes, i)
    #}}}

    # 'Overlap Studies': {{{
    for i in range(7,28):
        df['ta_DEMA_%d'%i] = talib.DEMA(closes, i)/closes
        df['ta_EMA_%d'%i]  = talib.EMA(closes,i)/closes
    df['ta_HT_TRENDLINE'] = talib.HT_TRENDLINE(closes)/closes
    for i in range(7,28):
        df['ta_KAMA_%d'%i] = talib.KAMA(closes, i)/closes
    #for c in [(2,28),(4,28)]:
    #    df['ta_MAVP_%d_%d'%(c[0],c[1])] = talib.MAVP(closes, 14, c[0],c[1])/closes
    for i in range(7,28):
        df['ta_MIDPOINT_%d'%i] = talib.MIDPOINT(closes, i)/closes
        df['ta_MIDPRICE_%d'%i] = talib.MIDPRICE(highs, lows,i)/closes
    df['ta_SAR'] = talib.SAR(highs,lows, 0.02, 0.2)/closes
    for i in range(7,28):
        df['ta_SMA_%d'%i] = talib.SMA(closes, i)/closes
    for i in range(2,21):
        df['ta_T3_%d'%i] = talib.T3(closes, i, 0.7)/closes
    for i in range(7,28):
        df['ta_TEMA_%d'%i] = talib.TEMA(closes, i)/closes
        df['ta_TRIMA_%d'%i] = talib.TRIMA(closes, i)/closes
        df['ta_WMA_%d'%i] = talib.WMA(closes, i)/closes

    # 'Pattern Recognition': [
    #df = cdl(df)

    # 'Volatility Indicators': [
    for i in [7,10,14,28]:
        #df['ta_ATR_%d'%i] = talib.ATR(highs, lows, closes, i)
        df['ta_NATR_%d'%i] = talib.NATR(highs, lows, closes, i)
    #df['ta_TRANGE'] = talib.TRANGE(highs, lows, closes)

    # 'Volume Indicators': [
    df['ta_AD'] = talib.AD(highs, lows, closes, volumes)/volumes
    for c in [(3,10), (6,20),(12,40)]:
        df['ta_ADOSC_%d_%d'%(c[0],c[1])] = talib.ADOSC(highs, lows, closes, volumes,c[0],c[1])/volumes
    df['ta_OBV'] = talib.OBV(closes, volumes)/volumes
    return df


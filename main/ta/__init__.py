#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author hongbin@youzan.com

# {{{ imports
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
# }}}


def diff(df, key, shink, shift=1):
    df["ta_diff_%s_%d_%d" % (key, shink, shift)] = (df[key]/df[key].shift(shift)).shift(shink)
    return df

def hdiff(df, key, hole):
    df["ta_hdiff_%s_%d"%(key, hole)] = df[key]/df[key].shift(hole)
    return df


def call1s3(df):
    opens =   df['open'].values
    highs =   df['high'].values
    lows =    df['low'].values
    closes =  df['close'].values
    volumes = df['volume'].values
    df = call1s1(df)
    for tri in [(6, 13),(12, 26), (24, 52)]:
        dfMACD = pta.MACD(df, tri[0],tri[1])
        dfMACD["ta_MACD1_%d_%d"%(tri[0],tri[1])] = dfMACD["ta_MACD_%d_%d"%(tri[0],tri[1])]/dfMACD["ta_MACDsign_%d_%d"%(tri[0],tri[1])]
        df = df.join(dfMACD["ta_MACD1_%d_%d"%(tri[0],tri[1])])
    return df

def call1s2(df):
    df = call1s1(df)
    df = df.set_index(['date'])    
    df['dual'] = 1
    dual = df['dual']
    dfDow = base.yeod('index_dow','^DJI')
    dfDow = dfDow.set_index(['date'])
    for i in range(0,10):
        dfDow = hdiff(dfDow, 'close', i)
        diff = dfDow['ta_hdiff_close_%d' % i]
        df2 = pd.concat([dual, diff],axis =1)
        df['ta_index_hdiff_close_%d' % i] = df2['ta_hdiff_close_%d' % i]
    df.reset_index(level=0, inplace = True)
    return df

def call1s130(df):
    df = call1(df)
    talist = ["date", "open", "high", "low", "close", "volume"]
    with open(os.path.join(root, 'data', 'models', 'model_tadowcall1_GBCv1n322md3lr001_l5_s1700e2009_importance')) as f:
        i = 0
        for line in f.readlines():
            tokens = line.split(",")
            feat = tokens[0].strip()
            talist.append(feat)
            i+=1
            if i > 30:
                break
    print "len of talist:", len(talist)
    df = df[talist]
    return df
def call1s1(df):
    df = call1(df)
    talist = ["date", "open", "high", "low", "close", "volume"]
    with open(os.path.join(root, 'data', 'models', 'model_tadowcall1_GBCv1n322md3lr001_l5_s1700e2009_importance')) as f:
        i = 0
        for line in f.readlines():
            tokens = line.split(",")
            feat = tokens[0].strip()
            talist.append(feat)
            i+=1
            if i > 106:
                break
    print "len of talist:", len(talist)
    df = df[talist]
    return df

def calltest1(df):
    df['open'] = df['open'] * 100
    df['high'] = df['high'] * 100
    df['low'] = df['low'] * 100
    df['close'] = df['close'] * 100
    df['volume'] = df['volume'] * 100
    df = call1(df)
    return df

def call1(df):
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes=df['close'].values
    volumes = df['volume'].values

    #Cycle Indicators': {{{
    df['ta_HT_DCPERIOD'] = talib.HT_DCPERIOD(closes)
    df['ta_HT_DCPHASE'] =  talib.HT_DCPHASE (closes)
    #df['ta_HT_PHASOR'] = talib.HT_PHASOR(closes)
    #df['ta_HT_SINE'] =   talib.HT_SINE(closes)
    #df['ta_HT_TRENDMODE'] = talib.HT_TRENDLINE(closes)
    #}}}

    #'Momentum Indicators': {{{
    for i in [7,10,14,28,52]:
        df['ta_ADX_%d' % i] = talib.ADX(highs, lows, closes, i)
        df['ta_ADXR_%d' % i] = talib.ADXR(highs, lows, closes, i)
    #for couple in [(6, 13), (12, 26), (24, 52)]:
    #    df['ta_APO_%d_%d' % (couple[0], couple[1])] = talib.APO(closes, couple[0], couple[1])
    #for i in range(7, 52):
    #    df['ta_AROON_%d' % i] = talib.AROON(highs, lows, i)
    #    df['ta_AROONOSC_%d' % i] = talib.AROONOSC(highs, lows, i)
    df['ta_BOP'] = talib.BOP(opens, highs, lows, closes)
    df['ta_CCI'] = talib.CCI(highs, lows, closes)
    for i in [7,10,14,28,52]:
        df['ta_CMO_%d' % i] = talib.CMO(closes, i)
        #df['ta_DX_%d' %i] = talib.DX(closes, i)
    #for tri in [(6,13,8),(12, 26, 9), (24, 52, 18)]:
    #    macd = talib.MACD(closes, tri[0], tri[1], tri[2])
    #    #df['ta_MACD_macd_%d_%d_%d'%(tri[0],tri[1],tri[2])] = macd[0]
    #    #df['ta_MACD_macdsignal_%d_%d_%d'%(tri[0],tri[1],tri[2])] = macd[1]
    #    #df['ta_MACD_macdhist_%d_%d_%d'%(tri[0],tri[1],tri[2])] = macd[2]
    #
    #for i in range(6,18):
    #    macd = talib.MACDFIX(closes,i)
    #    df['ta_MACDFIX_macd_%d'% i]= macd[0]
    #    df['ta_MACDFIX_macdsignal_%d'% i]= macd[1]
    #    df['ta_MACDFIX_macdhist_%d'% i]= macd[2]
    for i in [7,10,14,28,52]:
        df['ta_MFI_%d'%i] = talib.MFI(highs,lows,closes,volumes)
    for i in [7,10,14,28,52]:
        df['ta_MINUS_DI_%d'%i]  = talib.MINUS_DI(highs,lows,closes,i)
        df['ta_MINUS_DM_%d'%i]  = talib.MINUS_DI(highs,lows,closes,i)
        #df['ta_MOM_%d' % i]     = talib.MOM(closes, i)
        df['ta_PLUS_DI_%d' % i] = talib.PLUS_DI(highs,lows,closes,i)
        df['ta_PLUS_DM_%d' % i] = talib.PLUS_DM(highs,lows,i)
    for couple in [(6,13),(12,26),(25,52)]:
        df['ta_PPO_%d_%d'%(couple[0],couple[1])] = talib.PPO(closes, couple[0],couple[1])
    for i in [2,5,7,10,14,28,52]:
        df['ta_ROC_%d'% i] = talib.ROC(closes, i)
        df['ta_ROCP_%d'%i] = talib.ROCP(closes, i)
        df['ta_ROCR_%d'%i] = talib.ROCR(closes, i)
        df['ta_ROCR100_%d'%i] = talib.ROCR100(closes,i)
        df['ta_RSI_%d'%i] = talib.RSI(closes, i)
    for tri in [(5,3,3),(10,6,6),(20,12,12)]:
        stoch = talib.STOCH(highs,lows,closes, fastk_period=tri[0],slowk_period=tri[1],slowd_period=tri[2])
        df['ta_STOCH_slowk_%d_%d_%d' % (tri[0],tri[1],tri[2])] = stoch[0]
        df['ta_STOCH_slowd_%d_%d_%d' % (tri[0],tri[1],tri[2])] = stoch[1]
    for i in [2,5,7,10,14,28,52]:
        for c in [(5,3),(10,6),(20,12)]:
            stochrsi = talib.STOCHRSI(closes,i,c[0],c[1])
            df['ta_STOCHRSI_slowk_%d_%d_%d'%(i,c[0],c[1])] = stochrsi[0]
            df['ta_STOCHRSI_slowd_%d_%d_%d'%(i,c[0],c[1])] = stochrsi[1]
    for i in [2,5,7,10,14,28,52]:
        df['ta_TRIX_%d'%i] = talib.TRIX(closes,i)
    for tr in [(7,14,28),[14,28,52]]:
        df['ta_ULTOSC_%d_%d_%d'%(tr[0],tr[1],tr[2])] = talib.ULTOSC(highs,lows, closes, tr[0],tr[1],tr[2])
    for i in [2,5,7,10,14,28,52]:
        df['ta_WILLR_%d'%i] = talib.WILLR(highs, lows,closes, i)
    #}}}

    # 'Overlap Studies': {{{
    #for i in range(7,52):
        #df['ta_DEMA_%d'%i] = talib.DEMA(closes, i)
        #df['ta_EMA_%d'%i]  = talib.EMA(closes,i)
    #df['ta_HT_TRENDLINE'] = talib.HT_TRENDLINE(closes)
    #for i in range(7,52):
    #    df['ta_KAMA_%d'%i] = talib.KAMA(closes, i)
    #for c in [(2,30),(4,30),(4,50)]:
    #    df['ta_MAVP_%d_%d'%(c[0],c[1])] = talib.MAVP(closes, 14, c[0],c[1])
    #for i in range(7,52):
    #    df['ta_MIDPOINT_%d'%i] = talib.MIDPOINT(closes, i)
    #    df['ta_MIDPRICE_%d'%i] = talib.MIDPRICE(highs, lows,i)
    #df['ta_SAR'] = talib.SAR(highs,lows, 0.02, 0.2)
    #for i in range(7,52):
    #    df['ta_SMA_%d'%i] = talib.SMA(closes, i)
    #for i in range(2,21):
    #    df['ta_T3_%d'%i] = talib.T3(closes, i, 0.7)
    #for i in range(7,52):
    #    df['ta_TEMA_%d'%i] = talib.TEMA(closes, i)
    #    df['ta_TRIMA_%d'%i] = talib.TRIMA(closes, i)
    #    df['ta_WMA_%d'%i] = talib.WMA(closes, i)

    # 'Pattern Recognition': [
    df = cdl(df)

    # 'Volatility Indicators': [
    for i in [7,10,14,28,52]:
        #df['ta_ATR_%d'%i] = talib.ATR(highs, lows, closes, i)
        df['ta_NATR_%d'%i] = talib.NATR(highs, lows, closes, i)
    #df['ta_TRANGE'] = talib.TRANGE(highs, lows, closes)

    # 'Volume Indicators': [
    #df['ta_AD'] = talib.AD(highs, lows, closes, volumes)
    #for c in [(3,10), (6,20),(12,40)]:
    #    df['ta_ADOSC_%d_%d'%(c[0],c[1])] = talib.ADOSC(highs, lows, closes, volumes,c[0],c[1])
    #df['ta_OBV'] = talib.OBV(closes, volumes)
    return df
# }}}

def cdl(df): # {{{
    open = df.open.values
    high = df.high.values
    low = df.low.values
    close = df.close.values
    df["ta_cdl3blackcrows"] = talib.CDL3BLACKCROWS(open, high, low, close)
    df["ta_cdl3inside"] = talib.CDL3INSIDE(open, high, low, close)
    df["ta_cdl3linestrike"] = talib.CDL3LINESTRIKE(open, high, low, close)
    df["ta_cdl3outside"] = talib.CDL3OUTSIDE(open, high, low, close)
    df["ta_cdl3starsinsouth"] = talib.CDL3STARSINSOUTH(open, high, low, close)
    df["ta_cdl3whitesoldiers"] = talib.CDL3WHITESOLDIERS(open, high, low, close)
    df["ta_cdladandonedbaby"] = talib.CDLABANDONEDBABY(open, high, low, close) #? penetration
    df["ta_CDLADVANCEBLOCK"] = talib.CDLADVANCEBLOCK(open, high, low, close)
    df["ta_CDLBELTHOLD"] = talib.CDLBELTHOLD(open, high, low, close)
    df["ta_CDLBREAKAWAY"] = talib.CDLBREAKAWAY(open, high, low, close)
    df["ta_CDLCLOSINGMARUBOZU"] = talib.CDLCLOSINGMARUBOZU(open, high, low, close)
    df["ta_CDLCONCEALBABYSWALL"] = talib.CDLCONCEALBABYSWALL(open, high, low, close)
    df["ta_CDLCOUNTERATTACK"] = talib.CDLCOUNTERATTACK(open, high, low, close)
    df["ta_CDLDARKCLOUDCOVER"] = talib.CDLDARKCLOUDCOVER(open, high, low, close) #? penetration
    df["ta_CDLDARKCLOUDCOVER"] = talib.CDLDARKCLOUDCOVER(open, high, low, close) #? penetration
    df["ta_CDLDOJI"] = talib.CDLDOJI(open, high, low, close)
    df["ta_CDLDOJISTAR"] = talib.CDLDOJISTAR(open, high, low, close)
    df["ta_CDLDRAGONFLYDOJI"] = talib.CDLDRAGONFLYDOJI(open, high, low, close)
    df["ta_CDLENGULFING"] = talib.CDLENGULFING(open, high, low, close)
    df["ta_CDLEVENINGDOJISTAR"] = talib.CDLEVENINGDOJISTAR(open, high, low, close) #? penetration
    df["ta_CDLEVENINGSTAR"] = talib.CDLEVENINGSTAR(open, high, low, close) #? penetration
    df["ta_CDLGAPSIDESIDEWHITE"] = talib.CDLGAPSIDESIDEWHITE(open, high, low, close)
    df["ta_CDLGRAVESTONEDOJI"] = talib.CDLGRAVESTONEDOJI(open, high, low, close)
    df["ta_CDLHAMMER"] = talib.CDLHAMMER(open, high, low, close)
    df["ta_CDLHANGINGMAN"] = talib.CDLHANGINGMAN(open, high, low, close)
    df["ta_CDLHARAMI"] = talib.CDLHARAMI(open, high, low, close)
    df["ta_CDLHARAMICROSS"] = talib.CDLHARAMICROSS(open, high, low, close)
    df["ta_CDLHIGHWAVE"] = talib.CDLHIGHWAVE(open, high, low, close)
    df["ta_CDLHIKKAKE"] = talib.CDLHIKKAKE(open, high, low, close)
    df["ta_CDLHIKKAKEMOD"] = talib.CDLHIKKAKEMOD(open, high, low, close)
    df["ta_CDLHOMINGPIGEON"] = talib.CDLHOMINGPIGEON(open, high, low, close)
    df["ta_CDLIDENTICAL3CROWS"] = talib.CDLIDENTICAL3CROWS(open, high, low, close)
    df["ta_CDLINNECK"] = talib.CDLINNECK(open, high, low, close)
    df["ta_CDLINVERTEDHAMMER"] = talib.CDLINVERTEDHAMMER(open, high, low, close)
    df["ta_CDLKICKING"] = talib.CDLKICKING(open, high, low, close)
    df["ta_CDLKICKINGBYLENGTH"] = talib.CDLKICKINGBYLENGTH(open, high, low, close)
    df["ta_CDLLADDERBOTTOM"] = talib.CDLLADDERBOTTOM(open, high, low, close)
    df["ta_CDLLONGLEGGEDDOJI"] = talib.CDLLONGLEGGEDDOJI(open, high, low, close)
    df["ta_CDLLONGLINE"] = talib.CDLLONGLINE(open, high, low, close)
    df["ta_CDLMARUBOZU"] = talib.CDLMARUBOZU(open, high, low, close)
    df["ta_CDLMATCHINGLOW"] = talib.CDLMATCHINGLOW(open, high, low, close)
    df["ta_CDLMATHOLD"] = talib.CDLMATHOLD(open, high, low, close) #? penetration
    df["ta_CDLMORNINGDOJISTAR"] = talib.CDLMORNINGDOJISTAR(open, high, low, close) #? penetration
    df["ta_CDLMORNINGSTAR"] = talib.CDLMORNINGSTAR(open, high, low, close) #? penetration
    df["ta_CDLONNECK"] = talib.CDLONNECK(open, high, low, close)
    df["ta_CDLPIERCING"] = talib.CDLPIERCING(open, high, low, close)
    df["ta_CDLRICKSHAWMAN"] = talib.CDLRICKSHAWMAN(open, high, low, close)
    df["ta_CDLRISEFALL3METHODS"] = talib.CDLRISEFALL3METHODS(open, high, low, close)
    df["ta_CDLSEPARATINGLINES"] = talib.CDLSEPARATINGLINES(open, high, low, close)
    df["ta_CDLSHOOTINGSTAR"] = talib.CDLSHOOTINGSTAR(open, high, low, close)
    df["ta_CDLSHORTLINE"] = talib.CDLSHORTLINE(open, high, low, close)
    df["ta_CDLSPINNINGTOP"] = talib.CDLSPINNINGTOP(open, high, low, close)
    df["ta_CDLSTALLEDPATTERN"] = talib.CDLSTALLEDPATTERN(open, high, low, close)
    df["ta_CDLSTICKSANDWICH"] = talib.CDLSTICKSANDWICH(open, high, low, close)
    df["ta_CDLTAKURI"] = talib.CDLTAKURI(open, high, low, close)
    df["ta_CDLTASUKIGAP"] = talib.CDLTASUKIGAP(open, high, low, close)
    df["ta_CDLTHRUSTING"] = talib.CDLTHRUSTING(open, high, low, close)
    df["ta_CDLTRISTAR"] = talib.CDLTRISTAR(open, high, low, close)
    df["ta_CDLUNIQUE3RIVER"] = talib.CDLUNIQUE3RIVER(open, high, low, close)
    df["ta_CDLUPSIDEGAP2CROWS"] = talib.CDLUPSIDEGAP2CROWS(open, high, low, close)
    df["ta_CDLXSIDEGAP3METHODS"] = talib.CDLXSIDEGAP3METHODS(open, high, low, close)
    return df
# }}}

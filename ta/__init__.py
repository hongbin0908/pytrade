
#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author hongbin@youzan.com
import os,sys
import talib
local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..')
sys.path.append(root)

import strategy_mining.model_base as base

def diff(df, key, shink, shift=1):
    df["ta_diff_%s_%d_%d" % (key, shink, shift)] = (df[key]/df[key].shift(shift)).shift(shink)
    return df

def adx(df, timeperiod = 14):
    """
    ADX(high, low, close[, timeperiod=?])
    Average Directional Movement Index (Momentum Indicators)
    Inputs:
        prices: ['high', 'low', 'close']
    Parameters:
        timeperiod: 14
    Outputs:
        real
    """
    npAdx = talib.ADX(df.high.values, df.low.values, df.close.values, timeperiod)
    df["ta_adx_" + str(timeperiod)] = npAdx
    npAdxr = talib.ADXR(df.high.values, df.low.values, df.close.values, timeperiod)
    df["ta_adxr_" + str(timeperiod)] = npAdx
    npMdi = talib.MINUS_DI(df.high.values, df.low.values, df.close.values, timeperiod)
    df["ta_mdi_" + str(timeperiod)] = npMdi
    npMdi = talib.PLUS_DI(df.high.values, df.low.values, df.close.values, timeperiod)
    df["ta_pdi_" + str(timeperiod)] = npMdi
    return df

def apo(df, fastperiod=12, slowperiod=26, matype=0):
    """
    Absolute Price Oscillator (Momentum Indicators)

    """
    npApo = talib.APO(df.close.values, fastperiod, slowperiod, matype)
    df['ta_apo_%d_%d_%d'%(fastperiod, slowperiod, matype)] = npApo
    return df

def aroon(df, timeperiod=14):
    """
    Aroon (Momentum Indicators)
    """
    npUp, npDown = talib.AROON(df.high.values, df.low.values, timeperiod)
    df["ta_aroon_up_" + str(timeperiod)] = npUp 
    df["ta_aroon_down_" + str(timeperiod)] = npDown
    df['ta_aroonosc_' + str(timeperiod)] = talib.AROONOSC(df.high.values, df.low.values, timeperiod) 
    return df

def ad(df):
    """
    Chaikin A/D Line (Volume Indicators)
    """
    npAd = talib.AD(df.high.values, df.low.values, df.close.values, df.volume.values)
    df["ta_ad"] = npAd
    return df
def adosc(df, fastperiod = 3, slowperiod = 10):
    """
    Chaikin A/D Oscillator (Volume Indicators)
    """
    npAdosc = talib.ADOSC(df.high.values, df.low.values, df.close.values, df.volume.values, fastperiod, slowperiod)
    df["ta_adsoc"] = npAdosc
    return df

def obv(df, fastperiod = 3, slowperiod = 10):
    """
    On Balance Volume (Volume Indicators)
    """
    df["ta_obv"] = talib.OBV(df.close.values, df.volume.values)
    return df

def atr(df, timeperiod = 14):
    """
    Average True Range (Volatility Indicators)
    """
    df["ta_atr_" + str(timeperiod)] = talib.ATR(df.high.values, df.low.values, df.close.values, timeperiod)
    return df

def natr(df, timeperiod = 14):
    """
    Normalized Average True Range (Volatility Indicators)
    """
    df["ta_natr_" + str(timeperiod)] = talib.NATR(df.high.values, df.low.values, df.close.values, timeperiod)
    return df

def trange(df):
    """
    True Range (Volatility Indicators)
    """
    df["ta_trange"] = talib.TRANGE(df.high.values, df.low.values, df.close.values)
    return df


def cal_all(df):
    df = adx(df, timeperiod = 14)
    for i in range(14):
        df = diff(df, "ta_adx_14", i, 1)
    for i in range(14):
        df = diff(df, "ta_mdi_14", i, 1)
    for i in range(14):
        df = diff(df, "ta_pdi_14", i, 1)
    for i in range(14):
        df = diff(df, "close", i, 1)
    #df = diff(df, 0,1) 
    #df = diff(df, 1,1) 
    #df = diff(df, 2,1) 
    #df = diff(df, 3,1) 
    df = apo(df, 12, 26, 0)
    df = aroon(df, 14)
    df = ad(df)
    df = adosc(df)
    df = obv(df)
    df = atr(df, timeperiod = 14)
    df = natr(df, timeperiod = 14)
    df = trange(df)
    return df

if __name__ == '__main__':
    df = base.get_stock_data_pd("A")
    print cal_all(df).head(10)
    #print adx(df).tail(1)


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

def cdl(df):
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

def call2(df):
    df = cdl(df)
    return df

if __name__ == '__main__':
    df = base.get_stock_data_pd("A")
    print cal_all(df).head(10)
    #print adx(df).tail(1)

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

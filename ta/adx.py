#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author hongbin@youzan.com
import os,sys
import talib
local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..')
sys.path.append(root)

import strategy_mining.model_base as base



def cal(df, timeperiod = 14):
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
    df["ta_adx" + str(timeperiod)] = npAdx
    npMdi = talib.MINUS_DI(df.high.values, df.low.values, df.close.values, timeperiod)
    df["ta_mid" + str(timeperiod)] = npMdi
    npMdi = talib.PLUS_DI(df.high.values, df.low.values, df.close.values, timeperiod)
    df["ta_pid" + str(timeperiod)] = npMdi
    return df

if __name__ == '__main__':

    df = base.get_stock_data_pd("A")
    
    print adx(df).tail(1)

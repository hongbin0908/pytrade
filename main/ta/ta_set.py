#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author hongbin@youzan.com

import os,sys
import talib
import pandas as pd
import platform


if platform.platform().startswith("Windows"):
    TEST = True
elif platform.platform().startswith("Darwin"):
    TEST = True
else:
    TEST = False
local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..', '..')
sys.path.append(root)

import main.base as base
import main.pandas_talib.sig_123recall as sig_123recall
import main.pandas_talib.sig_adx as sig_adx
import main.pandas_talib.sig_upbreak as sig_upbreak
import main.pandas_talib.sig_ta_PLUS_DM_28 as sig_ta_PLUS_DM_28
import main.pandas_talib.sig_ta_WILLR_2_1 as sig_ta_WILLR_2_1
import main.pandas_talib.sig_ta_STOCHOSC_1 as sig_ta_STOCHOSC_1

class TaSet():
    def get_name(self):
        pass
    def get_ta(self, df, confer):
        pass
class TaSetSma1(TaSet):
    def get_name(self):
        return "base1"
    def sma_ratio(self,closes, first, second):
        ta_sma_first = talib.SMA(closes, first)
        ta_sma_second = talib.SMA(closes, second)
        res = ta_sma_second / ta_sma_first
        return res
    def get_ta(self, df, confer):
        opens = df['open'].values
        highs = df['high'].values
        lows = df['low'].values
        closes=df['close'].values
        volumes = df['volume'].values
        for i in range(5, 52):
            df["ta_sma_%d_%d" % (i, i+2)] = self.sma_ratio(closes, i, i+2)
        for i in range(5, 52):
            df["ta_sma_%d_%d" % (i, i+4)] = self.sma_ratio(closes, i, i+4)
        for i in range(5, 52):
            df["ta_sma_%d_%d" % (i, i+6)] = self.sma_ratio(closes, i, i+6)
        df = df.round(4)
        return df

class TaSetSma2(TaSet):
    def get_name(self):
        return "base1"
    def sma_ratio(self,closes, first, second, third):
        ta_sma_first = talib.SMA(closes, first)
        ta_sma_second = talib.SMA(closes, second)
        ta_sma_third = talib.SMA(closes, third)
        res1 = ta_sma_second / ta_sma_first
        res2 = ta_sma_third / ta_sma_second
        res = res2/res1
        return res
    def get_ta(self, df, confer):
        opens = df['open'].values
        highs = df['high'].values
        lows = df['low'].values
        closes=df['close'].values
        volumes = df['volume'].values
        for i in range(5, 52):
            df["ta_sma_%d_%d_%d" % (i, i+2, i+4)] = self.sma_ratio(closes, i, i+2,i+4)
        for i in range(5, 52):
            df["ta_sma_%d_%d_%d" % (i, i+4, i+8)] = self.sma_ratio(closes, i, i+4,i+8)
        for i in range(5, 52):
            df["ta_sma_%d_%d_%d" % (i, i+6, i+12)] = self.sma_ratio(closes, i, i+6,i+12)
        df = df.round(4)
        return df
class TaSetBase1(TaSet):
    def get_name(self):
        return "base1"
    def get_ta(self, df, confer):
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
        for i in [7,10,14,28]:
            df['ta_ADX_%d' % i] = talib.ADX(highs, lows, closes, i)
            df['ta_ADXR_%d' % i] = talib.ADXR(highs, lows, closes, i)
        #for couple in [(6, 13), (12, 26), (24, 52)]:
        #    df['ta_APO_%d_%d' % (couple[0], couple[1])] = talib.APO(closes, couple[0], couple[1])
        #for i in range(7, 52):
        #    df['ta_AROON_%d' % i] = talib.AROON(highs, lows, i)
        #    df['ta_AROONOSC_%d' % i] = talib.AROONOSC(highs, lows, i)
        df['ta_BOP'] = talib.BOP(opens, highs, lows, closes)
        df['ta_CCI'] = talib.CCI(highs, lows, closes)
        for i in [7,10,14,28]:
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
        for i in [7,10,14,28]:
            df['ta_MFI_%d'%i] = talib.MFI(highs,lows,closes,volumes)
        for i in [7,10,14,28]:
            df['ta_MINUS_DI_%d'%i]  = talib.MINUS_DI(highs,lows,closes,i)
            df['ta_MINUS_DM_%d'%i]  = talib.MINUS_DM(highs,lows,i)
            #df['ta_MOM_%d' % i]     = talib.MOM(closes, i)
            df['ta_PLUS_DI_%d' % i] = talib.PLUS_DI(highs,lows,closes,i)
            df['ta_PLUS_DM_%d' % i] = talib.PLUS_DM(highs,lows,i)

        for couple in [(6,13),(12,26)]:
            df['ta_PPO_%d_%d'%(couple[0],couple[1])] = talib.PPO(closes, couple[0],couple[1])
        for i in [2,5,7,10,14,28]:
            df['ta_ROC_%d'% i] = talib.ROC(closes, i)
            #df['ta_ROCP_%d'%i] = talib.ROCP(closes, i)
            #df['ta_ROCR_%d'%i] = talib.ROCR(closes, i)
            #df['ta_ROCR100_%d'%i] = talib.ROCR100(closes,i)
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
        for i in [28,30,32,40,56]:
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
        from main.ta import ta_cdl
        df = ta_cdl.main(df)

        # 'Volatility Indicators': [
        for i in [7,10,14,28]:
            #df['ta_ATR_%d'%i] = talib.ATR(highs, lows, closes, i)
            df['ta_NATR_%d'%i] = talib.NATR(highs, lows, closes, i)
        #df['ta_TRANGE'] = talib.TRANGE(highs, lows, closes)

        # 'Volume Indicators': [
        #df['ta_AD'] = talib.AD(highs, lows, closes, volumes)
        #for c in [(3,10), (6,20),(12,40)]:
        #    df['ta_ADOSC_%d_%d'%(c[0],c[1])] = talib.ADOSC(highs, lows, closes, volumes,c[0],c[1])
        #df['ta_OBV'] = talib.OBV(closes, volumes)
        df = df.round(4)
        return df


class TaSetBase1Ext4(TaSet):

    def get_name(self):
        return "TaBase1Ext4"

    def get_ta(self, df, confer):
        df = df.reset_index("date", drop = True)
        df = TaSetBase1().get_ta(df, confer)
        df = sig_123recall.main(df)
        df = sig_adx.main(df)
        df = sig_upbreak.main(df)
        df = sig_ta_PLUS_DM_28.main(df)
        del df["ta_PLUS_DM_28"]
        return df

class TaSetBase1Ext4El(TaSet):
    def get_name(self):
        return "TaBase1Ext4El"

    def get_ta(self, df, confer):
        df = df.reset_index("date", drop=True)
        df = TaSetBase1Ext4().get_ta(df, confer)
        ipt = os.path.join(root, "main/ta/feat_select_sp500Top50-base1_ext4")
        with open(ipt, "r") as f:
            for line in f.readlines():
                line = line.split()
                if "F" == line[1]:
                    if line[0] in df:
                        del df[line[0]]
                else:
                    if not line[0] in df:
                        pass
        return df

def get_sp500():
    df = pd.read_csv(os.path.join(root, "constituents-financials.csv"))
    df = df.sort_values("Market Cap", ascending=False)
    return [each["Symbol"].strip() for i, each in df.iterrows()]

class TaSetBase1Ext5(TaSet):
    def get_name(self):
        return "ta_base1_ext5"
    def get_ta(self, df):
        df = df.reset_index("date", drop = True)
        df = TaSetBase1().get_ta(df)
        df = sig_123recall.main(df)
        df = sig_adx.main(df)
        df = sig_upbreak.main(df)
        df = sig_ta_PLUS_DM_28.main(df)
        df = sig_ta_WILLR_2_1.main(df)
        df = sig_ta_STOCHOSC_1.main(df)
        del df["ta_PLUS_DM_28"]
        return df
class TaSetBase1Ext6(TaSet):
    def get_name(self):
        return "ta_base1_ext6"
    def get_ta(self, df):
        df = df.reset_index("date", drop = True)
        df = TaSetBase1Ext5().get_ta(df)
        for each in ["ta_ROC_5","ta_WILLR_2", "ta_ROC_7", "ta_WILLR_7", "ta_WILLR_5", "ta_STOCHRSI_slowd_28_5_3", "ta_NATR_28"]:
            for i in [1, 2, 5, 7, 14]:
                df["%s-shift-%d" % (each, i)] = df[each].shift(i)
        return df
class TaSetBase1Ext7(TaSet):
    def get_name(self):
        return "ta_base1_ext7"
    def get_ta(self, df):
        df = df.reset_index("date", drop = True)
        df = TaSetBase1Ext5().get_ta(df)
        for each in ["ta_ROC_5","ta_WILLR_2", "ta_ROC_7", "ta_WILLR_7", "ta_WILLR_5", "ta_STOCHRSI_slowd_28_5_3", "ta_NATR_28"]:
            for i in [1, 2, 5, 7, 14]:
                df["%s-shift-%d" % (each, i)] = df[each].shift(i)
                df["%s-level-%d" % (each, i)] = df[each] - df["%s-shift-%d" % (each, i)]
                del df["%s-shift-%d" % (each, i)]
        return df
class TaSetBase1Ext8(TaSet):
    def get_name(self):
        return "ta_base1_ext8"

    def get_ta(self, df):
        def call(row, pre, i):
            level = row["%s-level-%d" % (pre, i)]
            if level < 0:
                return row[pre] * 1 / ( 1 + abs(level))
            return row[pre] * (1 + abs(level))
        df = df.reset_index("date", drop = True)
        df = TaSetBase1Ext5().get_ta(df)
        for each in ["ta_ROC_5","ta_ROC_7", "ta_ROC_14", "ta_WILLR_7", "ta_WILLR_5", "ta_WILLR_2",
                     "ta_WILLR_14", "ta_STOCHRSI_slowd_28_5_3", "ta_NATR_28"]:
            assert isinstance(each, str)
            if not each.startswith("ta_"):
                continue
            for i in [1, 5, 14]:
                df["%s-shift-%d" % (each, i)] = df[each].shift(i)
                df["%s-level-%d" % (each, i)] = df[each] - df["%s-shift-%d" % (each, i)]
                df["%s-merge-%s-level-%s" % (each, each, i )] = df.apply(lambda row: call(row, each, i), axis = 1)
                del df["%s-shift-%d" % (each, i)]
                del df["%s-level-%d" % (each, i)]
        return df

if __name__ == "__main__":
    from run import run
    confer = run.getConf()
    df = pd.read_csv(os.path.join(root, "data", "yeod", "AAPL.csv"))
    df[['volume']] = df[["volume"]].astype(float)
    TaSetBaseExt4ELBit().get_ta(df, confer)

#-*-encoding:gbk-*-

import sys,os
local_path = os.path.dirname(os.path.abspath(sys.argv[0]))
sys.path.append(local_path + "/./")

from price_judgement import *
from two_crow_builder import *
from three_inside_pattern import *
from three_inside_strike import *
from other_pattern import *
from model_base import *
from volume_pattern import *
from cycle_pattern import *
import talib_ext as te
def build_features_1():
    """
    return : list. a list of feature builder class
    """
    feature_builder_list = []

    feature_builder_c(te.PRICE_9, feature_builder_list)
    feature_builder_c(te.PRICE_8, feature_builder_list)
    feature_builder_c(te.PRICE_7, feature_builder_list)
    feature_builder_c(te.PRICE_6, feature_builder_list)
    feature_builder_c(te.PRICE_5, feature_builder_list)
    feature_builder_c(te.PRICE_4, feature_builder_list)
    feature_builder_c(te.PRICE_3, feature_builder_list)
    feature_builder_c(te.PRICE_2, feature_builder_list)
    feature_builder_c(te.PRICE_1, feature_builder_list)
    feature_builder_c(te.PRICE_0, feature_builder_list)
    return feature_builder_list
def build_features():
    """
    return : list. a list of feature builder class
    """
    feature_builder_list = []

    three_inside_strike = three_inside_strike_builder()
    three_outside_move = three_outside_move_builder()
    three_star_south = three_star_south_builder()
    three_ad_white_soldier = three_ad_white_soldier_builder()
    abandoned_baby = abandoned_baby_builder()
    three_ad_block = three_ad_block_builder()
    belt_hold = belt_hold_builder()
    break_away = breakaway_builder()
    conceal_baby = conceal_baby_swallow_builder()

    #feature_builder(talib.CDLCOUNTERATTACK, feature_builder_list)    
    #feature_builder(talib.CDLDARKCLOUDCOVER , feature_builder_list)
    #feature_builder(talib.CDLDOJI , feature_builder_list)
    #feature_builder(talib.CDLDOJISTAR , feature_builder_list)
    #feature_builder(talib.CDLDRAGONFLYDOJI , feature_builder_list)
    #feature_builder(talib.CDLENGULFING , feature_builder_list)
    #feature_builder(talib.CDLEVENINGDOJISTAR , feature_builder_list)
    #feature_builder(talib.CDLEVENINGSTAR , feature_builder_list)
    #feature_builder(talib.CDLGAPSIDESIDEWHITE , feature_builder_list)
    #
    #feature_builder(talib.CDLGRAVESTONEDOJI  , feature_builder_list)
    #feature_builder(talib.CDLHAMMER  , feature_builder_list)
    #feature_builder(talib.CDLHANGINGMAN , feature_builder_list)
    #feature_builder(talib.CDLHARAMI , feature_builder_list)
    #feature_builder(talib.CDLHARAMICROSS , feature_builder_list)
    #feature_builder(talib.CDLHIGHWAVE , feature_builder_list)
    #feature_builder(talib.CDLHIKKAKE , feature_builder_list)
    #feature_builder(talib.CDLHIKKAKEMOD , feature_builder_list)
    #
    #
    #feature_builder(talib.CDLIDENTICAL3CROWS , feature_builder_list)
    #feature_builder(talib.CDLINNECK , feature_builder_list)
    #feature_builder(talib.CDLINVERTEDHAMMER , feature_builder_list)
    #feature_builder(talib.CDLKICKING , feature_builder_list)
    #feature_builder(talib.CDLKICKINGBYLENGTH , feature_builder_list)
    #feature_builder(talib.CDLLADDERBOTTOM , feature_builder_list)
    #feature_builder(talib.CDLLONGLEGGEDDOJI , feature_builder_list)
    #feature_builder(talib.CDLLONGLINE , feature_builder_list)
    #feature_builder(talib.CDLMARUBOZU , feature_builder_list)
    #feature_builder(talib.CDLMATCHINGLOW , feature_builder_list)
    #feature_builder(talib.CDLMATHOLD , feature_builder_list)
    #feature_builder(talib.CDLMORNINGDOJISTAR , feature_builder_list)
    #feature_builder(talib.CDLMORNINGSTAR , feature_builder_list)
    #feature_builder(talib.CDLONNECK , feature_builder_list)
    #feature_builder(talib.CDLPIERCING , feature_builder_list)
    #
    #feature_builder(talib.CDLRICKSHAWMAN , feature_builder_list)
    #feature_builder(talib.CDLRISEFALL3METHODS , feature_builder_list)
    #feature_builder(talib.CDLSEPARATINGLINES , feature_builder_list)
    #feature_builder(talib.CDLSHOOTINGSTAR , feature_builder_list)
    #feature_builder(talib.CDLSHORTLINE , feature_builder_list)
    #feature_builder(talib.CDLSPINNINGTOP , feature_builder_list)
    #feature_builder(talib.CDLSTALLEDPATTERN , feature_builder_list)
    #
    #feature_builder(talib.CDLSTICKSANDWICH , feature_builder_list)
    #feature_builder(talib.CDLTAKURI , feature_builder_list)
    #feature_builder(talib.CDLTASUKIGAP , feature_builder_list)
    #feature_builder(talib.CDLTHRUSTING , feature_builder_list)
    #feature_builder(talib.CDLTRISTAR , feature_builder_list)
    #feature_builder(talib.CDLUNIQUE3RIVER , feature_builder_list)
    #feature_builder(talib.CDLUPSIDEGAP2CROWS , feature_builder_list)
    #
    #feature_builder(talib.CDLXSIDEGAP3METHODS , feature_builder_list)

    feature_builder_ohc(talib.ADXR, feature_builder_list)
    feature_builder_ohc(talib.CCI, feature_builder_list)
    feature_builder_ohc(talib.MINUS_DI, feature_builder_list)
    feature_builder_ohc(talib.PLUS_DI, feature_builder_list)
    feature_builder_ohc(talib.WILLR, feature_builder_list)
    # add by hongbin begin
    feature_builder_ohc(te.ADX_ext1, feature_builder_list)
    feature_builder_ohc(te.ADX_ext2, feature_builder_list)
    feature_builder_ohc(te.ADX_ext3, feature_builder_list)
    feature_builder_ohc(te.ADX_ext4, feature_builder_list)
    feature_builder_ohc(te.ADX_ext5, feature_builder_list)
    feature_builder_ohc(te.ADX_ext10, feature_builder_list)
    #feature_builder_ohc(te.ADX_ext30, feature_builder_list)
    feature_builder_ohc(te.ADX_ext201, feature_builder_list)
    feature_builder_ohc(te.ADX_ext202, feature_builder_list)
    feature_builder_ohc(te.ADX_ext203, feature_builder_list)
    feature_builder_ohc(te.ADX_ext204, feature_builder_list)
    feature_builder_ohc(te.ADX_ext205, feature_builder_list)
    feature_builder_ohc(te.ADX_ext210, feature_builder_list)
    feature_builder_ohc(te.ADX_ext301, feature_builder_list)
    feature_builder_ohc(te.ADX_ext302, feature_builder_list)
    feature_builder_ohc(te.ADX_ext303, feature_builder_list)
    feature_builder_ohc(te.ADX_ext304, feature_builder_list)
    feature_builder_ohc(te.ADX_ext305, feature_builder_list)
    feature_builder_ohc(te.ADX_ext310, feature_builder_list)
    feature_builder_ohc(te.ADX_ext401, feature_builder_list)
    feature_builder_ohc(te.ADX_ext402, feature_builder_list)
    feature_builder_ohc(te.ADX_ext403, feature_builder_list)
    feature_builder_ohc(te.ADX_ext404, feature_builder_list)
    feature_builder_ohc(te.ADX_ext405, feature_builder_list)
    feature_builder_ohc(te.ADX_ext410, feature_builder_list)
    feature_builder_c(te.MACD_ext101, feature_builder_list)
    feature_builder_c(te.MACD_ext102, feature_builder_list)
    feature_builder_c(te.MACD_ext103, feature_builder_list)
    feature_builder_c(te.MACD_ext104, feature_builder_list)
    feature_builder_c(te.MACD_ext105, feature_builder_list)
    feature_builder_c(te.EMA_1, feature_builder_list)
    feature_builder_c(te.EMA_101, feature_builder_list)
    feature_builder_c(te.EMA_102, feature_builder_list)
    feature_builder_c(te.EMA_103, feature_builder_list)
    feature_builder_c(te.EMA_104, feature_builder_list)
    feature_builder_c(te.EMA_105, feature_builder_list)
    feature_builder_c(te.EMA_110, feature_builder_list)
    feature_builder_c(te.EMA_2, feature_builder_list)
    feature_builder_c(te.EMA_201, feature_builder_list)
    feature_builder_c(te.EMA_202, feature_builder_list)
    feature_builder_c(te.EMA_203, feature_builder_list)
    feature_builder_c(te.EMA_204, feature_builder_list)
    feature_builder_c(te.EMA_205, feature_builder_list)
    feature_builder_c(te.EMA_210, feature_builder_list)
    feature_builder_c(te.EMA_3, feature_builder_list)
    feature_builder_c(te.EMA_301, feature_builder_list)
    feature_builder_c(te.EMA_302, feature_builder_list)
    feature_builder_c(te.EMA_303, feature_builder_list)
    feature_builder_c(te.EMA_304, feature_builder_list)
    feature_builder_c(te.EMA_305, feature_builder_list)
    feature_builder_c(te.EMA_310, feature_builder_list)
    feature_builder_c(te.RSI, feature_builder_list)
    feature_builder_c(te.RSI_101, feature_builder_list)
    feature_builder_c(te.RSI_102, feature_builder_list)
    feature_builder_c(te.RSI_103, feature_builder_list)
    feature_builder_c(te.RSI_104, feature_builder_list)
    feature_builder_c(te.RSI_105, feature_builder_list)
    feature_builder_c(te.RSI_110, feature_builder_list)
    
    #feature_builder_ohc(te.ADX_ext230, feature_builder_list)
    # add by hongbin end
    feature_builder_volume(talib.ADOSC, feature_builder_list)
    feature_builder_volume(talib.AD, feature_builder_list)
    feature_builder_ohc(talib.ATR, feature_builder_list)
    feature_builder_ohc(talib.NATR, feature_builder_list)
    feature_builder_ohc(talib.TRANGE, feature_builder_list)
    feature_builder_c(talib.HT_DCPERIOD, feature_builder_list)
    feature_builder_c(talib.HT_DCPHASE, feature_builder_list)
    feature_builder_c(talib.EMA  , feature_builder_list)
    feature_builder_c(talib.HT_TRENDLINE, feature_builder_list)
    feature_builder_c(talib.MA, feature_builder_list)
    feature_builder_c(talib.LINEARREG, feature_builder_list)
    feature_builder_c(talib.STDDEV, feature_builder_list)
    feature_builder_c(talib.TSF, feature_builder_list)
    feature_builder_c(talib.VAR, feature_builder_list)
#    feature_builder_list.append(three_inside_strike)
#    feature_builder_list.append(three_outside_move)
#    feature_builder_list.append(three_star_south)
#    feature_builder_list.append(three_ad_white_soldier)
#    feature_builder_list.append(abandoned_baby)
#    feature_builder_list.append(three_ad_block)
#    feature_builder_list.append(belt_hold)
#    feature_builder_list.append(break_away)
#    feature_builder_list.append(conceal_baby)    

    return feature_builder_list

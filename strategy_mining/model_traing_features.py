#-*-encoding:gbk-*-

import sys,os
local_path = os.path.dirname(os.path.abspath(sys.argv[0]))
sys.path.append(local_path + "/./")

from price_judgement import *
from two_crow_builder import *
from three_inside_pattern import *
from three_inside_strike import *
from other_pattern import *
from momentum_pattern import *
from volume_pattern import *

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

    feature_builder(talib.CDLCOUNTERATTACK, feature_builder_list)    
    feature_builder(talib.CDLDARKCLOUDCOVER , feature_builder_list)
    feature_builder(talib.CDLDOJI , feature_builder_list)
    feature_builder(talib.CDLDOJISTAR , feature_builder_list)
    feature_builder(talib.CDLDRAGONFLYDOJI , feature_builder_list)
    feature_builder(talib.CDLENGULFING , feature_builder_list)
    feature_builder(talib.CDLEVENINGDOJISTAR , feature_builder_list)
    feature_builder(talib.CDLEVENINGSTAR , feature_builder_list)
    feature_builder(talib.CDLGAPSIDESIDEWHITE , feature_builder_list)
    
    feature_builder(talib.CDLGRAVESTONEDOJI  , feature_builder_list)
    feature_builder(talib.CDLHAMMER  , feature_builder_list)
    feature_builder(talib.CDLHANGINGMAN , feature_builder_list)
    feature_builder(talib.CDLHARAMI , feature_builder_list)
    feature_builder(talib.CDLHARAMICROSS , feature_builder_list)
    feature_builder(talib.CDLHIGHWAVE , feature_builder_list)
    feature_builder(talib.CDLHIKKAKE , feature_builder_list)
    feature_builder(talib.CDLHIKKAKEMOD , feature_builder_list)
    
    
    feature_builder(talib.CDLIDENTICAL3CROWS , feature_builder_list)
    feature_builder(talib.CDLINNECK , feature_builder_list)
    feature_builder(talib.CDLINVERTEDHAMMER , feature_builder_list)
    feature_builder(talib.CDLKICKING , feature_builder_list)
    feature_builder(talib.CDLKICKINGBYLENGTH , feature_builder_list)
    feature_builder(talib.CDLLADDERBOTTOM , feature_builder_list)
    feature_builder(talib.CDLLONGLEGGEDDOJI , feature_builder_list)
    feature_builder(talib.CDLLONGLINE , feature_builder_list)
    feature_builder(talib.CDLMARUBOZU , feature_builder_list)
    feature_builder(talib.CDLMATCHINGLOW , feature_builder_list)
    feature_builder(talib.CDLMATHOLD , feature_builder_list)
    feature_builder(talib.CDLMORNINGDOJISTAR , feature_builder_list)
    feature_builder(talib.CDLMORNINGSTAR , feature_builder_list)
    feature_builder(talib.CDLONNECK , feature_builder_list)
    feature_builder(talib.CDLPIERCING , feature_builder_list)
    
    feature_builder(talib.CDLRICKSHAWMAN , feature_builder_list)
    feature_builder(talib.CDLRISEFALL3METHODS , feature_builder_list)
    feature_builder(talib.CDLSEPARATINGLINES , feature_builder_list)
    feature_builder(talib.CDLSHOOTINGSTAR , feature_builder_list)
    feature_builder(talib.CDLSHORTLINE , feature_builder_list)
    feature_builder(talib.CDLSPINNINGTOP , feature_builder_list)
    feature_builder(talib.CDLSTALLEDPATTERN , feature_builder_list)
    
    feature_builder(talib.CDLSTICKSANDWICH , feature_builder_list)
    feature_builder(talib.CDLTAKURI , feature_builder_list)
    feature_builder(talib.CDLTASUKIGAP , feature_builder_list)
    feature_builder(talib.CDLTHRUSTING , feature_builder_list)
    feature_builder(talib.CDLTRISTAR , feature_builder_list)
    feature_builder(talib.CDLUNIQUE3RIVER , feature_builder_list)
    feature_builder(talib.CDLUPSIDEGAP2CROWS , feature_builder_list)
    
    feature_builder(talib.CDLXSIDEGAP3METHODS , feature_builder_list)
    feature_builder_ohc(talib.ADXR, feature_builder_list)
    feature_builder_ohc(talib.CCI, feature_builder_list)
    feature_builder_ohc(talib.MINUS_DI, feature_builder_list)
    feature_builder_ohc(talib.PLUS_DI, feature_builder_list)
    feature_builder_ohc(talib.WILLR, feature_builder_list)
    feature_builder_volume(talib.ADOSC, feature_builder_list)
    feature_builder_volume(talib.AD, feature_builder_list)
    feature_builder_list.append(three_inside_strike)
    feature_builder_list.append(three_outside_move)
    feature_builder_list.append(three_star_south)
    feature_builder_list.append(three_ad_white_soldier)
    feature_builder_list.append(abandoned_baby)
    feature_builder_list.append(three_ad_block)
    feature_builder_list.append(belt_hold)
    feature_builder_list.append(break_away)
    feature_builder_list.append(conceal_baby)    

    return feature_builder_list

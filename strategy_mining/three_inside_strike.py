#-*-encoding:gbk-*-
import sys
import os
import talib
import numpy
import two_crow_builder
class three_inside_strike_builder():
    def feature_build(self, open_price, high_price, low_price, close_price, adjust_close, volume, index, feature_result_list):
        result = talib.CDL3LINESTRIKE(open_price, high_price, low_price, close_price)
        return result
class three_outside_move_builder():
    def feature_build(self, open_price, high_price, low_price, close_price, adjust_close, volume, index, feature_result_list):
        result = talib.CDL3OUTSIDE(open_price, high_price, low_price, close_price)
        return result 
class three_star_south_builder():
    def feature_build(self, open_price, high_price, low_price, close_price, adjust_close, volume, index, feature_result_list):
        result = talib.CDL3STARSINSOUTH(open_price, high_price, low_price, close_price)
        return result

class three_ad_white_soldier_builder():
    def feature_build(self, open_price, high_price, low_price, close_price, adjust_close, volume, index, feature_result_list):
        result = talib.CDL3WHITESOLDIERS(open_price, high_price, low_price, close_price)
        return result

class abandoned_baby_builder():
    def feature_build(self, open_price, high_price, low_price, close_price, adjust_close, volume, index, feature_result_list):
        result = talib.CDLABANDONEDBABY(open_price, high_price, low_price, close_price)
        return result

class three_ad_block_builder():
    def feature_build(self, open_price, high_price, low_price, close_price, adjust_close, volume, index, feature_result_list):
        result = talib.CDLADVANCEBLOCK(open_price, high_price, low_price, close_price)
        return result

class belt_hold_builder():
    def feature_build(self, open_price, high_price, low_price, close_price, adjust_close, volume, index, feature_result_list):
        result = talib.CDLBELTHOLD(open_price, high_price, low_price, close_price)
        return result
    
class breakaway_builder():
    def feature_build(self, open_price, high_price, low_price, close_price, adjust_close, volume, index, feature_result_list):
        result = talib.CDLBREAKAWAY(open_price, high_price, low_price, close_price)
        return result

class conceal_baby_swallow_builder():
    def feature_build(self, open_price, high_price, low_price, close_price, adjust_close, volume, index, feature_result_list):
        result = talib.CDLCONCEALBABYSWALL(open_price, high_price, low_price, close_price)
        return result
if __name__ == "__main__":
    print "begin"
    tmp_low = []
    tmp_high = []
    tmp_open = []
    tmp_close = []
    tmp_volume = []
    tmp_adjust = []
    two_crow_builder.load_data(sys.argv[1], tmp_open, tmp_high, tmp_low, tmp_close, tmp_adjust, tmp_volume)

    low_price = numpy.array(tmp_low) 
    high_price = numpy.array(tmp_high)
    open_price = numpy.array(tmp_open)
    close_price = numpy.array(tmp_close)
    adjust_close = numpy.array(tmp_adjust)
    volume_list = numpy.array(tmp_volume)
    feature_result_list = [-1]
    builder = three_inside_up_builder()
    builder.feature_build(open_price, high_price, low_price, close_price, adjust_close, volume_list, 0, feature_result_list)

#-*-encoding:gbk-*-
import sys
import os
import talib
import numpy
import two_crow_builder
class feature_builder():
    def __init__(self, feature_func, builder_list):
        self.feature_build_func = feature_func
        builder_list.append(self)
    def feature_build(self, open_price, high_price, low_price, close_price, adjust_close, volume, index, feature_result_list):
        result = self.feature_build_func(open_price, high_price, low_price, close_price)
        return result
    def feature_build_with_three_price(self, open_price, high_price, low_price, close_price, adjust_close, volume, index, feature_result_list):
        result = self.feature_build_func(open_price, high_price, close_price)
        middle_index = len(open_price)-1
        feature_result_list[index] = result[middle_index]
            
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
    builder_list = []
    feature_builder(talib.CDLBREAKAWAY, builder_list)
    feature_builder(talib.CDLCOUNTERATTACK, builder_list)
    print len(builder_list)
    for s in builder_list:
        s.feature_build(open_price, high_price, low_price, close_price, adjust_close, volume_list, 0, feature_result_list)

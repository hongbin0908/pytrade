#-*-encoding:gbk-*-
import sys
import os
import talib
import numpy
import random
class twocrow_builder():
    def feature_build(self, open_price, high_price, low_price, close_price, adjust_price, volume, index, feature_result_list):
        result = talib.CDL2CROWS(open_price, high_price, low_price, close_price)
        return result
            
if __name__ == "__main__":
    print "begin"
    tmp_low = []
    tmp_high = []
    tmp_open = []
    tmp_close = []
    tmp_adjust = []
    tmp_volume = []
    load_data(sys.argv[1], tmp_open, tmp_high, tmp_low, tmp_close, tmp_adjust, tmp_volume)

    low_price = numpy.array(tmp_low) 
    high_price = numpy.array(tmp_high)
    open_price = numpy.array(tmp_open)
    close_price = numpy.array(tmp_close)
    adjust_price = numpy.array(tmp_adjust)
    volume_list = numpy.array(tmp_volume)
    feature_result_list = [None]
    builder = twocrow_builder()
    builder.feature_build(open_price, high_price, low_price, close_price, adjust_price, volume_list, 0, feature_result_list)

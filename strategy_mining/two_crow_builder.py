#-*-encoding:gbk-*-
import sys
import os
import talib
import numpy
class twocrow_builder():
    def feature_build(self, open_price, high_price, low_price, close_price, adjust_price, volume, index, feature_result_list):
        result = talib.CDL2CROWS(open_price, high_price, low_price, close_price)
        middle_index = len(open_price)-1
        feature_result_list[index] = result[middle_index]

def load_data(filename, open_price, high_price, low_price, close_price, adjust_price, volume_list):
    fd = open(filename, "r")
    for j in fd:
        try:
            line_list = j.rstrip().split(",")
            open_p = float(line_list[1])
            high_p = float(line_list[2])
            low_p = float(line_list[3])
            close_p = float(line_list[4])
            open_price.append(open_p)
            high_price.append(high_p)
            low_price.append(low_p)
            close_price.append(close_p)
            volume = int(line_list[5])
            adjust_p = float(line_list[6])
            adjust_price.append(adjust_p)
            volume_list.append(volume)
        except Exception, e:
            continue
    fd.close()
            
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

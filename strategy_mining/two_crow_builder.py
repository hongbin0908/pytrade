#-*-encoding:gbk-*-
import sys
import os
import talib
import numpy
class twocrow_builder():
    def feature_build(self, open_price, high_price, low_price, close_price, index, feature_result_list):
        result = talib.CDL2CROWS(open_price, high_price, low_price, close_price)
        middle_index = len(open_price)/2 + 1
        feature_result_list[index] = result[middle_index]

def load_data(filename, open_price, high_price, low_price, close_price):
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
        except Exception, e:
            continue
    fd.close()
            
if __name__ == "__main__":
    print "begin"
    tmp_low = []
    tmp_high = []
    tmp_open = []
    tmp_close = []
    load_data(sys.argv[1], tmp_open, tmp_high, tmp_low, tmp_close)

    low_price = numpy.array(tmp_low) 
    high_price = numpy.array(tmp_high)
    open_price = numpy.array(tmp_open)
    close_price = numpy.array(tmp_close)
    feature_result_list = []
    builder = twocrow_builder()
    builder.feature_build(open_price, high_price, low_price, close_price, 9, feature_result_list)

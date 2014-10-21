#-*-encoding:gbk-*-
import sys
import os
import model_training_features

class feature_basic_extracotr:
    def process(self, stock_loader_result):
        extractor_result = []
        builder = model_straining_features.builder()
        for f in stock_loader_result:
            tmp_list_date = []
            tmp_list_begin = []
            tmp_list_high = []
            tmp_list_low = []
            tmp_list_close = []
            tmp_list_volume = []
            tmp_list_adjust_close = []
            for m in stock_loader_result[f]:
                tmp_list_date.append(m[0])
                tmp_list_begin.append(float(m[1]))
                tmp_list_high.append(float(m[2]))
                tmp_list_low.append(float(m[3]))
                tmp_list_close.append(float(m[4]))
                tmp_list_volume.append(float(m[5]))
                tmp_list_adjust_close.append(float(m[6]))
            
            array_date = numpy.array(tmp_list_date)
            array_begin = numpy.array(tmp_list_begin)
            array_high = numpy.array(tmp_list_high)
            array_low = numpy.array(tmp_list_low)
            array_close = numpy.array(tmp_list_close)
            array_volume = numpy.array(tmp_list_volume)
            array_adjust = numpy.array(tmp_list_adjust_close)
            tmp_results = []
            for s in builder:
                tmp_result = s.build(tmp_list_high, tmp_list_low, tmp_list_close, tmp_list_volume, tmp_list_adjust_close)
                tmp_results.append(tmp_result)
            tmp_list = numpy.hstack(array_date, tmp_result)
            extractor_result[f] = tmp_list
                

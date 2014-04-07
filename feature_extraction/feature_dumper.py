#-*-encoding:gbk-*-
import sys
import os
import numpy

class feature_basic_dumper:
    def process(self, dumper_result, dumper_file):
        for m in dumper_result:
            tmp_str = m
            for f in dumper_result[m]:
                for s in f:
                    tmp_str = tmp_str + str(s)
                print "%s" %(tmp_str)

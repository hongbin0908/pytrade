#!/usr/bin/env python
#@author binblue@126.com
import sys,os
local_path = os.path.dirname(os.path.abspath(sys.argv[0]))
sys.path.append(local_path + "/./")

import numpy 
import math
class prices_judgement:
    def judge(self, opens, highs, lows, closes, threshold, window):
        judge_list = numpy.array(closes)
        judge_list.fill(numpy.NaN)
        for index, s in enumerate(closes):
            if index < window or index > len(closes) - window:
                continue
            b_sum = sum(closes[index-window:index])
            a_sum = sum(closes[index+1: index+window+1])
            var = numpy.var(closes[index-window:index])
            judge_list[index] = a_sum*1.0/b_sum
#            if a_sum*1.0/b_sum > 1 + threshold:
#                judge_list[index] = 1
#            elif a_sum*1.0/b_sum < 1 - threshold :
#                judge_list[index] = 0
#            else:
#                judge_list[index] = -2
#
        return judge_list

class prices_judgement2:
    def judge(self, opens, highs, lows, closes, threshold, window):
        judge_list = numpy.array(closes)
        judge_list.fill(numpy.NaN)
        for index, s in enumerate(closes):
            if index < window or index > len(closes) - window:
                continue
            b_sum = sum(closes[index-window:index])
            a_sum = sum(closes[index+1: index+window+1])
            var = numpy.var(closes[index-window:index])
            if a_sum*1.0 > b_sum :
                judge_list[index] = 1
            elif a_sum <= b_sum :
                judge_list[index] = 0
        return judge_list


if __name__ == "__main__":
    judgeer = prices_judgement()
    price_list = [1, 2, 3, 4, 5]
    result = judgeer.judge(price_list, 0.05, 2)
    print result
    price_list = [1, 1, 1, 1, 1]
    result = judgeer.judge(price_list, 0.05, 2)
    print result

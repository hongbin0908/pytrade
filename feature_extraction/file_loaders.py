#-*-encoding:gbk-*-
import sys
import os
import numpy as np
import datetime
class file_basic_loader:
    def process(self, stock_name_file, stock_dir, begin_str, end_str):
        tmp_map = {}
        fd = open(stock_name_file, "r")
        begin_date = datetime.datetime.strptime(begin_str, "%Y-%m-%d")
        end_date = datetime.datetime.strptime(end_str, "%Y-%m-%d")
        for s in fd:
            stock_name = s.rstrip()
            stock_file = stock_dir + "/" + stock_name
            try:
                stock_fd = open(stock_file, "r")
            except Exception, e:
                sys.stderr.write("exception:info=%s" %(e))
                continue
            tmp_list = []
            num = 0
            for m in stock_fd:
                line_list = m.rstrip().split(",")
                if num == 0:
                    num = 1
                    continue
                date_curr = datetime.datetime.strptime(line_list[0], "%Y-%m-%d")
                if date_curr > end_date or date_curr < begin_date:
                    break
                tmp_list.append(line_list)
            tmp_list.reverse()
            if len(tmp_list) > 0:
                tmp_map[stock_name] = tmp_list
            return tmp_map


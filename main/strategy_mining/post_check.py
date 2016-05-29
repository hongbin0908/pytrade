#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author binred@outlook.com
 desc: 

����Ԥ�������Ʊ���ݽ���"�º����". ��������ʵ�ļ۸�, ��Ϊ��Ԥ��Ķ���. ������һЩͳ������.
	
	���ڻع�ģ��, ����Ԥ��ǰ10����Ʊ���ǵ���ȷ��/R2score����.
	���ڷ���ģ��, ����Ԥ���ǰ10����Ʊ�ǵ���ȷ��.
����	data/prices_series/YYYY-MM-DD/yesterday.csv
	�����Ԥ���Ʊ����
	/home/work/workplace/stock_data/
	��Ʊ�����ݿ�
���	data/prices_series/YYYY-MM-DD/yesterday_post.csv
	�����Ԥ������, ��������ʵ�ļ۸�
	data/prices_series/YYYY-MM-DD/yesterday_post.csv.ana
	��ǰ��Ԥ�����ݵķ���
	���ڻع�ģ��, ����Ԥ��ǰ10����Ʊ���ǵ���ȷ��/R2score����.
	���ڷ���ģ��, ����Ԥ���ǰ10����Ʊ�ǵ���ȷ��.

"""

import sys,os
import datetime
from optparse import OptionParser
import  model_base as base

local_path = os.path.dirname(os.path.abspath(sys.argv[0]))
sys.path.append(local_path + "/./")

def readinput(directory, stockdir):
    assert(directory != None)
    filename = os.path.join(directory, "predicted.csv")
    if not os.path.exists(filename):
        filename = os.path.join(directory, "predicted.cv")

        if not os.path.exists(filename):
            print "can not find file name : %s" % (filename)
    postfile = open(os.path.join(directory, "predicted_post.csv"), "w")
    for line in open(filename, "r"):
         terms = line.rstrip().split(",")
         if not  len(terms) == 4:
             print "terms is not 4 splited:%s" % line
             sys.exit(1)
         sym = terms[0]
         strdate1 = terms[1].strip(); assert(len(strdate1) == 10)
         span = int(terms[2])
         predicted = terms[3]

        
         stock_prices1 = base.get_stock_data_one_day(sym, strdate1, stockdir)
         stock_prices2 = base.get_stock_data_span_day(sym, strdate1, span, stockdir)
         assert("close" in stock_prices1)
         assert("close" in stock_prices2)

         print >> postfile, "%s,%s,%d,%s,%s,%s" % (sym, strdate1, span, stock_prices2["date"], predicted, stock_prices2["close"] * 1.0 /stock_prices1["close"] * 10000)
    postfile.close()



def main(options, args):
    readinput(options.direct, options.stockdir)

    
def parse_options(parser): #{{{
    """
    parse command line
    """
    parser.add_option("-V", "--version", dest="version", action = "store_true",
                      default=False, help = "if show the version of alitta")
    parser.add_option("-v", "--verbose",
                      action = "store_true", dest = "verbose",
                      default=True, help = "make lots of noise[default]")
    parser.add_option("-d", "--direct", dest="direct", type="string", 
            default=None, action="store", help="the input/output file directionary")
    parser.add_option("-s", "--stockdir", dest="stockdir", type="string", 
            default="/home/work/workplace/stock_data/", action="store", help="the stock data directionary")
    return parser.parse_args()
# }}}


if __name__ == '__main__':
    parser = OptionParser()
    (options, args) = parse_options(parser)
    main(options, args)

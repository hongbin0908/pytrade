#!/usr/bin/env python
#@author redbin@outlook.com

"""
generate the price series of stocks which base on the first day's price
"""
import sys,os
local_path = os.path.dirname(os.path.abspath(sys.argv[0]))
sys.path.append(local_path + "/./")
# parse the command paramaters
from optparse import OptionParser

from model_base import get_file_list,get_stock_data,get_stock_from_path
import logging

import subprocess


logging.basicConfig(level = logging.DEBUG,
    format = '%(levelname)s %(asctime)s [%(filename)s:%(lineno)d] %(message)s',                                                                              
    filename = 'log/%s.log' % (os.path.basename(sys.argv[0]),), 
    filemode = 'a')
def main(options, args):
    #cmd_str = "mkdir -p " + options.output
    #cmd = subprocess.Popen("ls")
    #cmd.wait()
    options.window = int(options.window)

    f_train = open(options.output + "/" + "train.csv", "w")
    f_last = open(options.output + "/" + "last.csv", "w")

    file_list = get_file_list("/home/work/workplace/stock_data/")
    stock_num = 0
    for f in file_list:
        stock_num += 1
        if stock_num % 10 == 0:
            logging.debug("build the %d's stock" % stock_num)
        symbol = get_stock_from_path(f)
        dates, open_prices, high_prices, low_prices, close_prices, adjust_close_prices, volumes = get_stock_data(f)
        if len(dates)  < 499:
            logging.debug("%s is too short(%d)!" % (symbol, len(dates)))
            continue
        for i in range(len(close_prices)-options.window-1):
            for  j in range(options.window):
                inc = close_prices[i+j+1]/close_prices[i+j]
                print >> f_train, "%f," % inc,
            classes = 0
            if close_prices[i+options.window + 1] > close_prices[i+options.window] :
                classes = 1
            print >> f_train, "%d" % classes
        print >> f_last, "%s," % dates[-1],
        print >> f_last, "%s," % symbol,
        for i in range(len(close_prices)-options.window-1, len(close_prices)-1):
            inc = close_prices[i+1]/close_prices[i]
            if i != (len(close_prices)-2):
                print >> f_last, "%f," % inc ,
            else:
                print >> f_last, "%f" % inc 
    f_train.close()
    f_last.close()



def parse_options(paraser):
    """
    parser command line
    """
    parser.add_option("--window", dest="window",action = "store", default=60, help = "the history price window")
    parser.add_option("--output", dest="output",action = "store", default="data/prices_series/", help = "the output directory")
    return parser.parse_args()

if __name__ == "__main__":
    parser = OptionParser()
    (options, args) = parse_options(parser)
    main(options, args)

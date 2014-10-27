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
    filename = local_path + '/log/%s.log' % (os.path.basename(sys.argv[0]),), 
    filemode = 'a')
def extract_features_and_classes(close_prices, window): #{{{
    ret = ""
    for i in range(len(close_prices)-window-1):
        for  j in range(window):
            inc = close_prices[i+j+1] * 1.0 / close_prices[i+j]
            ret +=  str(inc) + ","
        classes = 0
        if close_prices[i+window + 1] > close_prices[i+window] :
             classes = 1
        ret += "%d" % classes + "\n"
    return ret
# }}}

def extract_last_features(symbol, dates, close_prices, window): #{{{
    assert(len(dates) == len(close_prices))
    ret = ""
    ret += symbol + ","
    ret += str(dates[-1]) + ","
    for i in range(len(close_prices)-window-1, len(close_prices)-1):
        inc = close_prices[i+1]*1.0/close_prices[i]
        if i != (len(close_prices)-2):
            ret += str(inc) + "," 
        else:
            ret += str(inc) + "\n" 
    return ret
# }}}

def main(options, args): # {{{
    #cmd_str = "/bin/mkdir -p " + options.output
    #print cmd_str
    #cmd = subprocess.Popen(cmd_str)
    #cmd.wait()
    options.window = int(options.window)

    f_train = open(options.output + "/" + "train.csv", "w")
    f_last = open(options.output + "/" + "last.csv", "w")

    file_list = get_file_list(options.stocks_path)
    stock_num = 0
    for f in file_list:
        stock_num += 1
        if stock_num % 10 == 0:
            logging.debug("build the %d's stock" % stock_num)
        symbol = get_stock_from_path(f)
        dates, open_prices, high_prices, low_prices, close_prices, adjust_close_prices, volumes = get_stock_data(f)
        if len(dates)  < options.limit:
            logging.debug("%s is too short(%d)!" % (symbol, len(dates)))
            continue
        print >> f_train, "%s" %  \
            extract_features_and_classes(close_prices, options.window),
        print >> f_last, "%s" % \
                extract_last_features(symbol, dates, close_prices, options.window),
    f_train.close()
    f_last.close()
# }}}

def parse_options(parser): #{{{
    """
    parser command line
    """
    parser.add_option("--window", dest="window",action = "store", \
            default=60, help = "the history price window")
    parser.add_option("--output", dest="output",action = "store", \
            default=local_path + "/data/prices_series/", help = "the output directory")
    parser.add_option("--stocks_path", dest="stocks_path",action = "store", \
            default="/home/work/workplace/stock_data/", \
            help = "the stocks data directory")
    parser.add_option("--limit", type="int", dest="limit",action = "store", \
            default=499, \
            help = "the limit length of stock data")
    return parser.parse_args()
# }}}

if __name__ == "__main__": #{{{
    parser = OptionParser()
    (options, args) = parse_options(parser)
    main(options, args)
# }}}

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

class ExtractorBase: # {{{
    def __init__(self, symbol, dates, open_prices, high_prices, low_prices, close_prices, window, volumes=None):
        self.symbol = symbol
        self.dates = dates
        self.open_prices = open_prices
        self.high_prices = high_prices
        self.low_prices = low_prices
        self.close_prices = close_prices
        self.window = window
        self.volumes = volumes
    def extract_features_and_classes(self):
        assert(False)
    def extract_yesterday_features_and_classes(self):
        assert(False)
    def extract_last_features(self):
        assert(False)
# }}}
class Extractor1(ExtractorBase):
    def extract_features_and_classes(self): #{{{
        ret = ""
        for i in range(len(self.close_prices)-self.window-1):
            for  j in range(self.window):
                inc = self.close_prices[i+j+1] * 1.0 / self.close_prices[i+j]
                ret +=  str(inc) + ","
            classes = 0
            if self.close_prices[i+self.window + 1] > self.close_prices[i+self.window] :
                 classes = 1
            ret += "%d" % classes + "\n"
        return ret
    # }}}

    def extract_last_features(self): #{{{
        assert(len(self.dates) == len(self.close_prices))
        ret = ""
        ret += self.symbol + ","
        ret += str(self.dates[-1]) + ","
        for i in range(len(self.close_prices)-self.window-1, len(self.close_prices)-1):
            inc = self.close_prices[i+1]*1.0/self.close_prices[i]
            if i != (len(self.close_prices)-2):
                ret += str(inc) + "," 
            else:
                ret += str(inc) + "\n" 
        return ret
    # }}}
        
class Extractor2(ExtractorBase):
    def extract_features_and_classes(self): #{{{
        ret = ""
        for i in range(len(self.close_prices)-self.window-1):
            for  j in range(self.window):
                inc = self.close_prices[i+j+1] * 1.0 / self.close_prices[i+0]
                ret +=  str(inc) + ","
            classes = 0
            if self.close_prices[i+self.window + 1] > self.close_prices[i+self.window] :
                 classes = 1
            ret += "%d" % classes + "\n"
        return ret
    # }}}

    def extract_last_features(self): #{{{
        assert(len(self.dates) == len(self.close_prices))
        ret = ""
        ret += self.symbol + ","
        ret += str(self.dates[-1]) + ","
        for i in range(len(self.close_prices)-self.window-1, len(self.close_prices)-1):
            inc = self.close_prices[i+1]*1.0/self.close_prices[len(self.close_prices)-self.window-1]
            if i != (len(self.close_prices)-2):
                ret += str(inc) + "," 
            else:
                ret += str(inc) + "\n" 
        return ret
    # }}}

class Extractor3(ExtractorBase):
    def extract_features_and_classes(self): #{{{
        ret = ""
        for i in range(len(self.close_prices)-self.window-1):
            for  j in range(self.window):
                inc = self.open_prices[i+j+1] * 1.0 / self.close_prices[i]
                
                ret +=  str(inc) + ","
                inc = self.high_prices[i+j+1] * 1.0 / self.close_prices[i]
                
                ret +=  str(inc) + ","
                inc = self.low_prices[i+j+1] * 1.0 / self.close_prices[i]
                
                ret +=  str(inc) + ","
                inc = self.close_prices[i+j+1] * 1.0 / self.close_prices[i+0]
                ret +=  str(inc) + ","
            classes = 0
            if self.close_prices[i+self.window + 1] > self.close_prices[i+self.window] :
                 classes = 1
            ret += "%d" % classes + "\n"
        return ret
    # }}}

    def extract_last_features(self): #{{{
        assert(len(self.dates) == len(self.close_prices))
        ret = ""
        ret += self.symbol + ","
        ret += str(self.dates[-1]) + ","
        for i in range(len(self.close_prices)-self.window-1, len(self.close_prices)-1):
            inc = self.open_prices[i+1]*1.0/self.close_prices[len(self.close_prices)-self.window-1]
            ret += str(inc) + "," 
            inc = self.high_prices[i+1]*1.0/self.close_prices[len(self.close_prices)-self.window-1]
            ret += str(inc) + "," 
            inc = self.low_prices[i+1]*1.0/self.close_prices[len(self.close_prices)-self.window-1]
            ret += str(inc) + "," 
            inc = self.close_prices[i+1]*1.0/self.close_prices[len(self.close_prices)-self.window-1]
            if i != (len(self.close_prices)-2):
                ret += str(inc) + "," 
            else:
                ret += str(inc) + "\n" 
        return ret
    # }}}

class Extractor4(ExtractorBase):
    def extract_features_and_classes(self): #{{{
        ret = ""
        for i in range(len(self.close_prices)-self.window-1):
            for  j in range(self.window):
                inc = self.open_prices[i+j+1] * 1.0 / self.close_prices[i+j]
                inc = int(inc * 10000) 
                
                ret +=  str(inc) + ","
                inc = self.high_prices[i+j+1] * 1.0 / self.close_prices[i+j]
                inc = int(inc * 10000) 
                
                ret +=  str(inc) + ","
                inc = self.low_prices[i+j+1] * 1.0 / self.close_prices[i+j]
                inc = int(inc * 10000) 
                
                ret +=  str(inc) + ","
                inc = self.close_prices[i+j+1] * 1.0 / self.close_prices[i+j]
                inc = int(inc * 10000) 
                ret +=  str(inc) + ","
            classes = 0
            if self.close_prices[i+self.window + 1] > self.close_prices[i+self.window] :
                 classes = 1
            ret += "%d" % classes + "\n"
        return ret
    # }}}


    def extract_yesterday_features_and_classes(self): #{{{
        assert(len(self.dates) == len(self.close_prices))
        ret = ""
        ret += self.symbol + ","
        ret += str(self.dates[-1]) + ","
        for i in range(len(self.close_prices)-self.window-2, len(self.close_prices)-2):
            inc = self.open_prices[i+1]*1.0/self.close_prices[i]
            inc = int(inc * 10000) 
            ret += str(inc) + "," 
            inc = self.high_prices[i+1]*1.0/self.close_prices[i]
            inc = int(inc * 10000) 
            ret += str(inc) + "," 
            inc = self.low_prices[i+1]*1.0/self.close_prices[i]
            inc = int(inc * 10000) 
            ret += str(inc) + "," 
            inc = self.close_prices[i+1]*1.0/self.close_prices[i]
            inc = int(inc * 10000) 
            ret += str(inc) + "," 
        clazz = 0
        if self.close_prices[-1] > self.close_prices[-2]:
            clazz = 1
        else:
            clazz = 0
        ret += str(clazz) + "\n"
        return ret
    # }}}

    def extract_last_features(self): #{{
        assert(len(self.dates) == len(self.close_prices))
        ret = ""
        ret += self.symbol + ","
        ret += str(self.dates[-1]) + ","
        for i in range(len(self.close_prices)-self.window-1, len(self.close_prices)-1):
            inc = self.open_prices[i+1]*1.0/self.close_prices[i]
            inc = int(inc * 10000) 
            ret += str(inc) + "," 
            inc = self.high_prices[i+1]*1.0/self.close_prices[i]
            inc = int(inc * 10000) 
            ret += str(inc) + "," 
            inc = self.low_prices[i+1]*1.0/self.close_prices[i]
            inc = int(inc * 10000) 
            ret += str(inc) + "," 
            inc = self.close_prices[i+1]*1.0/self.close_prices[i]
            inc = int(inc * 10000) 
            if i != (len(self.close_prices)-2):
                ret += str(inc) + "," 
            else:
                ret += str(inc) + "\n" 
        return ret
    # }}}
class Extractor5(ExtractorBase):
    def extract_features_and_classes(self): #{{{
        ret = ""
        for i in range(len(self.close_prices)-self.window-1):
            for  j in range(self.window):
                inc = self.open_prices[i+j+1] * 1.0 / self.close_prices[i+j]
                inc = int(inc * 10000) 
                ret +=  str(inc) + ","

                inc = self.high_prices[i+j+1] * 1.0 / self.close_prices[i+j]
                inc = int(inc * 10000) 
                ret +=  str(inc) + ","

                inc = self.low_prices[i+j+1] * 1.0 / self.close_prices[i+j]
                inc = int(inc * 10000) 
                ret +=  str(inc) + ","

                inc = self.close_prices[i+j+1] * 1.0 / self.close_prices[i+j]
                inc = int(inc * 10000) 
                ret +=  str(inc) + ","

                inc = self.volumes[i+j+1] * 1.0 / self.volumes[i+j]
                inc = int(inc * 10000)
                ret += str(inc) + ","

            classes = 0
            if self.close_prices[i+self.window + 1] > self.close_prices[i+self.window] :
                 classes = 1
            ret += "%d" % classes + "\n"
        return ret
    # }}}


    def extract_yesterday_features_and_classes(self): #{{{
        assert(len(self.dates) == len(self.close_prices))
        ret = ""
        ret += self.symbol + ","
        ret += str(self.dates[-1]) + ","
        for i in range(len(self.close_prices)-self.window-2, len(self.close_prices)-2):
            inc = self.open_prices[i+1]*1.0/self.close_prices[i]
            inc = int(inc * 10000) 
            ret += str(inc) + "," 
            inc = self.high_prices[i+1]*1.0/self.close_prices[i]
            inc = int(inc * 10000) 
            ret += str(inc) + "," 
            inc = self.low_prices[i+1]*1.0/self.close_prices[i]
            inc = int(inc * 10000) 
            ret += str(inc) + "," 
            inc = self.close_prices[i+1]*1.0/self.close_prices[i]
            inc = int(inc * 10000) 
            ret += str(inc) + "," 
            inc = self.volumes[i+1] * 1.0 / self.volumes[i]
            inc = int(inc * 10000)
            ret += str(inc) + ","
        clazz = 0
        if self.close_prices[-1] > self.close_prices[-2]:
            clazz = 1
        else:
            clazz = 0
        ret += str(clazz) + "\n"
        return ret
    # }}}

    def extract_last_features(self): #{{
        assert(len(self.dates) == len(self.close_prices))
        ret = ""
        ret += self.symbol + ","
        ret += str(self.dates[-1]) + ","
        for i in range(len(self.close_prices)-self.window-1, len(self.close_prices)-1):
            inc = self.open_prices[i+1]*1.0/self.close_prices[i]
            inc = int(inc * 10000) 
            ret += str(inc) + "," 
            inc = self.high_prices[i+1]*1.0/self.close_prices[i]
            inc = int(inc * 10000) 
            ret += str(inc) + "," 
            inc = self.low_prices[i+1]*1.0/self.close_prices[i]
            inc = int(inc * 10000) 
            ret += str(inc) + "," 
            inc = self.volumes[i+1] * 1.0 / self.volumes[i]
            inc = int(inc * 10000)
            ret += str(inc) + ","
            inc = self.close_prices[i+1]*1.0/self.close_prices[i]
            inc = int(inc * 10000) 
            if i != (len(self.close_prices)-2):
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
    f_yesterday = open(options.output + "/" + "yesterday.csv", "w")

    # get the extractor
    Extractor = globals()[options.extractor]
    file_list = get_file_list(options.stocks_path)
    stock_num = 0
    for f in file_list:
        stock_num += 1
        if stock_num % 10 == 0:
            logging.debug("build the %d's stock" % stock_num)
        symbol = get_stock_from_path(f)
        dates, open_prices, high_prices, low_prices, close_prices, adjust_close_prices, volumes = get_stock_data(f)
        extractor = Extractor(symbol, dates, open_prices, high_prices, low_prices, close_prices, options.window, volumes)
        if len(dates)  < options.limit:
            logging.debug("%s is too short(%d)!" % (symbol, len(dates)))
            continue
        print >> f_train, "%s" %  \
            extractor.extract_features_and_classes(),
        print >> f_last, "%s" % \
                extractor.extract_last_features(),
        print >> f_yesterday, "%s" % \
                extractor.extract_yesterday_features_and_classes(),
    f_train.close()
    f_last.close()
    f_yesterday.close()
# }}}

def parse_options(parser): #{{{
    """
    parser command line
    """
    parser.add_option("--extractor", dest="extractor",action = "store", \
            default="Extractor5", help = "the extractor to use")
    parser.add_option("--window", type="int", dest="window",action = "store", \
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

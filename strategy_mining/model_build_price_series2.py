#!/usr/bin/env python
#@author redbin@outlook.com

"""
generate the price series of stocks which base on the first day's price
"""
import sys,os,json
local_path = os.path.dirname(os.path.abspath(sys.argv[0]))
sys.path.append(local_path + "/./")
# parse the command paramaters
from optparse import OptionParser

from model_base import get_file_list,get_stock_data,get_stock_from_path,get_date_str
import logging

import subprocess
from model_extractor_base import ExtractorBase
from model_extractor4 import Extractor4


filedir = local_path + '/log/'
if not os.path.exists(filedir):
    os.makedirs(filedir)
logging.basicConfig(level = logging.DEBUG,
    format = '%(levelname)s %(asctime)s [%(filename)s:%(lineno)d] %(message)s',
    filename = filedir + '%s.log' % (os.path.basename(sys.argv[0]),), 
    filemode = 'a')

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


def main(options, args): # {{{
    #cmd_str = "/bin/mkdir -p " + options.output
    #print cmd_str
    #cmd = subprocess.Popen(cmd_str)
    #cmd.wait()
    options.window = int(options.window)

    if options.utildate == None:
        print "ERROR: the utildate is NONE!"
        assert(False)
    d_train     = options.output + "/" + options.extractor+  "_" + str(options.span) + "/" + options.utildate ;
    if not os.path.exists(d_train) : os.makedirs(d_train)
    f_train     = open(d_train + "/features.csv", "w")
    d_last      = options.output + "/" + options.extractor+  "_" + str(options.span) + "/" + options.utildate  ;
    if not os.path.exists(d_train) : os.makedirs(d_last)
    f_last      = open( d_last + "/last.csv", "w")
    f_meta      = open( d_train + "/meta.json", "w")
    # get the extractor
    Extractor = globals()[options.extractor]
    file_list = get_file_list(options.stocks_path)
    stock_num = 0
    for f in file_list:
        stock_num += 1
        if stock_num % 10 == 0:
            logging.debug("build the %d's stock" % stock_num)
        symbol = get_stock_from_path(f)
        dates, open_prices, high_prices, low_prices, close_prices, adjust_close_prices, volumes = get_stock_data(f, options.utildate, 360)
        if len(dates)  < options.limit:
            logging.debug("%s is too short(%d)!" % (symbol, len(dates)))
            continue
        if not dates[-1] == options.utildate:
            print "the last date(%s) of stock data is not %s of symbol(%s)" % (dates[-1], options.utildate,symbol)
            continue
        extractor = Extractor(symbol, dates, open_prices, high_prices, low_prices, close_prices, options.window, options.span, options.isregress, volumes)
        print >> f_train, "%s" %  \
            extractor.extract_features_and_classes(),
        print >> f_last, "%s" % \
                extractor.extract_last_features(),
    
    f_train.close()
    f_last.close()
    meta = {}
    meta["span"] = options.span
    meta["extractor"] = options.extractor
    meta["limit"] = options.limit

    print >> f_meta, "%s" %  json.dumps(meta)
# }}}

def parse_options(parser): #{{{
    """
    parser command line
    """
    parser.add_option("--extractor", dest="extractor",action = "store", \
            default="Extractor4", help = "the extractor to use")
    parser.add_option("--window", type="int", dest="window",action = "store", \
            default=60, help = "the history price window")
    parser.add_option("--output", dest="output",action = "store", \
            default=local_path + "/data/prices_series/", help = "the output directory")
    parser.add_option("--stocks_path", dest="stocks_path",action = "store", \
            default="/home/work/workplace/stock_data/", \
            help = "the stocks data directory")
    parser.add_option("--limit", type="int", dest="limit",action = "store", \
            default=360, \
            help = "the limit length of stock data")
    parser.add_option("--utildate", dest="utildate",action = "store", default=None, help = "the last date to train")
    parser.add_option("--span", dest="span",action = "store", type="int", default=5,help = "the span of the price to predict")
    parser.add_option("--isregress", dest="isregress",action = "store_true", default=True, help = "using repgress model or classify?")
    return parser.parse_args()
# }}}

if __name__ == "__main__": #{{{
    parser = OptionParser()
    (options, args) = parse_options(parser)
    main(options, args)
# }}}

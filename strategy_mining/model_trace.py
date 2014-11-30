#!/usr/bin/env python
#@author redbin@outlook.com

"""
once the model_tuner.py generate the predict.csv , it time to check whether the model 
is good or bad.
"""
import sys,os
local_path = os.path.dirname(os.path.abspath(sys.argv[0]))
sys.path.append(local_path + "/./")
from optparse import OptionParser

def parse_options(parser): #{{{
    """
    parser command line
    """
    parser.add_option("--input", dest="input",action = "store", \
            default=local_path + "/data/prices_series/", help = "the output directory")
    parser.add_option("--utildate", dest="utildate",action = "store", default=None, help = "the last date to train")
    parser.add_option("--stocks_path", dest="stocks_path",action = "store", \
            default="/home/work/workplace/stock_data/", \
            help = "the stocks data directory")
    return parser.parse_args()
# }}}


def readfile(filename): # {{{

# }}}

def main(options, args): # {{{
    if options.utildate = None:
        options.utildate = get_date_str(1)
    predictfile = options.input + "/" + options.utildate + "/" + "predict.csv"

# }}}

if __name__ == "__main__": #{{{
    parser = OptionParser()
    (options, args) = parse_options(parser)
    main(options, args)
# }}}

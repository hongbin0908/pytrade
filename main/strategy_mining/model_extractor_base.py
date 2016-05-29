#!/usr/bin/env python
#@author binred@outlook.com

"""
generate the price series of stocks which base on the first day's price
"""
import sys,os
local_path = os.path.dirname(os.path.abspath(sys.argv[0]))
sys.path.append(local_path + "/./")
class ExtractorBase: # {{{
    def __init__(self, symbol, dates, open_prices, high_prices, low_prices, close_prices, window, span, isregress, volumes=None):
        self.symbol = symbol
        self.dates = dates
        self.open_prices = open_prices
        self.high_prices = high_prices
        self.low_prices = low_prices
        self.close_prices = close_prices
        self.window = window
        self.span = span
        self.isregress = isregress
        self.volumes = volumes
    def extract_features_and_classes(self):
        assert(False)
    def extract_last_features(self):
        assert(False)
# }}}

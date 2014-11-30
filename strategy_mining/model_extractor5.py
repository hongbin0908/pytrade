#!/usr/bin/env python
#@author redbin@outlook.com

"""
generate the price series of stocks which base on the first day's price
"""
import sys,os
local_path = os.path.dirname(os.path.abspath(sys.argv[0]))
sys.path.append(local_path + "/./")
from model_extractor_base import ExtractorBase

class Extractor5(ExtractorBase):
    def extract_features_and_classes(self): #{{{
        ret = ""
        for i in range(len(self.close_prices)-self.window-self.span):
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


                inc = self.volumes[i+j+1] * 1.0 / self.volumes[i+j]
                inc = int(inc * 10000)
                ret += str(inc) + ","

                inc = self.close_prices[i+j+1] * 1.0 / self.close_prices[i+j]
                inc = int(inc * 10000) 
                ret +=  str(inc) + ","

            if self.isregress == False:
                classes = 0
                if self.close_prices[i+self.window + self.span] > self.close_prices[i+self.window] :
                     classes = 1
                ret += "%d" % classes + "\n"
            else :
                ret += "%d" % ((self.close_prices[i+self.window + self.span] * 1.0  / self.close_prices[i+self.window]) * 10000) + "\n"

        return ret
    # }}}

    def extract_last_features(self): #{{{
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

#!/usr/bin/env python2.7
# author hongbin0908@126.com
# a strategy called extend broke up
# is a short term long method from "HIT and Run"

import sys,os,time
local_path = os.path.dirname(os.path.abspath(sys.argv[0]))
sys.path.append(local_path + "/./")
sys.path.append(local_path + "/../common")

import numpy
import talib
import pyalgotrade.talibext.indicator
from pyalgotrade import strategy
from pyalgotrade.barfeed import yahoofeed
from pyalgotrade.tools import yahoofinance
from pyalgotrade import plotter
from utils import *
from multibacktesting import *


class ExtBrokeUpStrategy(Multibacktesting):
    """
    """
    def __init__(self):
        Multibacktesting.__init__(self)
    def long(self, instrument, datetime, barDs):
        entry = EntryInfo() 

        #adx = pyalgotrade.talibext.indicator.ADX(barDs, 100,14)
        highs = barDs.getHighDataSeries()
        closes = barDs.getCloseDataSeries()
        opens = barDs.getOpenDataSeries()
        lows = barDs.getLowDataSeries()
        if len(highs)  < 240:
            return entry
        cur_range = closes[-1] - opens[-1]
        for i in range(30):
            price_range = highs[-1 -i -1] - lows[-1 -i -1]
            if (cur_range < price_range):
                return entry
        for i in range(239):
            if (highs[-1] < highs[-1 - i - 1]):
                return entry
        entry.entryType = 1
        return entry
    def short(self, instrument, datetime, barDs):
        entry = EntryInfo()
        return entry
    def exit(self,instrument, position, datetime, barDs):
        exit = ExitInfo()
        highs = barDs.getHighDataSeries()
        closes = barDs.getCloseDataSeries()
        lows = closes = barDs.getLowDataSeries()
        cur_high = highs[-1]
        cur_close = closes[-1]
        cur_low = lows[-1]
        entry_price = position.get_entry_price()
        if cur_close > entry_price * 1.10:
            cur_exit =  position.getCurExitOrder() 
            if cur_exit == None or cur_exit.stopPrice < cur_close * 0.95:
                exit.exitType = 1
                exit.stopPrice = cur_close * 0.95
        else:
            entime = position.get_entry_time()
            enprice = position.get_entry_price()
            delta = time.mktime(datetime.timetuple()) - time.mktime(entime.timetuple()) 
            if cur_high < enprice * 0.98  and  delta > 10 * 24 * 3600:
                exit.exitType = 1
            elif cur_close < entry_price * 0.95 :
                exit.exitType = 1
        return exit

if __name__ == '__main__':
    ExtBrokeUpStrategy().run_plot()

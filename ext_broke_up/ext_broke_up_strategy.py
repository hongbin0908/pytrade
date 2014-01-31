#!/usr/bin/env python2.7
# author hongbin0908@126.com
# a strategy called extend broke up
# is a short term long method from "HIT and Run"

import sys,os
local_path = os.path.dirname(os.path.abspath(sys.argv[0]))
sys.path.append(local_path + "/./")

import numpy
import talib
import pyalgotrade.talibext.indicator
from pyalgotrade import strategy
from pyalgotrade.barfeed import yahoofeed
from pyalgotrade.tools import yahoofinance
from pyalgotrade import plotter


class ExtBrokeUpStrategy(strategy.BacktestingStrategy):
    """
    __instrument : the code trading
    """
    def __init__(self, feed, instruments):
        strategy.BacktestingStrategy.__init__(self, feed)
        self.__instruments = instruments
        self.__positions = {}
        self.__stops= {}
        for instrument in instruments:
            self.__positions[instrument] = None
            self.__stops[instrument] = -1
    def entry_signal(self, barDs):
        #adx = pyalgotrade.talibext.indicator.ADX(barDs, 100,14)
        highs = barDs.getHighDataSeries()
        lows = barDs.getLowDataSeries()
        if len(highs)  < 30:
            return False
        cur_range = highs[-1] - lows[-1]
        for i in range(9):
            price_range = highs[-1 -i -1] - lows[-1 -i -1]
            if (cur_range < price_range):
                return False
        for i in range(29):
            if (highs[-1] < highs[-1 - i - 1]):
                return False
        return True
    def exit_signal(self,instrument, barDs):
        position = self.__positions[instrument]
        highs = barDs.getHighDataSeries()
        closes = barDs.getCloseDataSeries()
        lows = closes = barDs.getLowDataSeries()
        cur_high = highs[-1]
        cur_close = closes[-1]
        cur_low = lows[-1]
        entry_price = position.getEntryOrder().getExecutionInfo().getPrice()
        if (self.__stops[instrument] < 0) :
            self.__stops[instrument] = cur_close * 0.95
        elif cur_low < self.__stops[instrument]:
            return True
        else:
            if self.__stops[instrument] < cur_close * 0.95:
                self.__stops[instrument] = cur_close * 0.95
        return False

    def onBars(self, bars):
        for instrument in self.__instruments:
            barDs = self.getFeed().getDataSeries(instrument)
            if self.__positions[instrument] == None:
                if self.entry_signal(barDs):
                    self.__positions[instrument] = self.enterLong(instrument, 10, True)
            else:
                if self.exit_signal(instrument, barDs):
                    self.__positions[instrument].exit()
                    entryOrder = self.__positions[instrument].getEntryOrder().getExecutionInfo()
                    exitOrder = self.__positions[instrument].getExitOrder().getExecutionInfo()
                    print "trade: buy:%f %s sell:%f %s" % (entryOrder.getPrice(), entryOrder.getDateTime(),0,0)# exitOrder.getPrice(), exitOrder.getDateTime())
                    self.__positions[instrument] = None

def main():
    instruments = ["orcl","bidu"]
    feed = yahoofinance.build_feed(instruments, 2011,2012,".")
    stg = ExtBrokeUpStrategy(feed, instruments)
    plt = plotter.StrategyPlotter(stg, True, True, True)
    stg.run()
    plt.plot()


if __name__ == '__main__':
    main()

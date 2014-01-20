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


class ExtBrokeUpStrategy(strategy.BacktestingStrategy):
    """
    __instrument : the code trading
    """
    def __init__(self, feed, instrument):
        strategy.BacktestingStrategy.__init__(self, feed)
        self.__instrument = instrument
        self.__position = None
    def onBars(self, bars):
        barDs = self.getFeed().getDataSeries("orcl")
        adx = pyalgotrade.talibext.indicator.ADX(barDs, 100,14)
        if adx == None:
            return
        print adx[-1]
        if adx[-1] > 10:
            self.__position = self.enterLong(self.__instrument, 10, True)
        if adx[-1] < 10:
            if (self.__position != None):
                self.__position.exit()
    def onEnterOk(self, position):
        execInfo = position.getEntryOrder().getExecutionInfo()
        print "%s: BUY at $%.2f" % (execInfo.getDateTime(), execInfo.getPrice())
    def onEnterCanceled(self, position):
        self.__position = None
    def onExitOk(self, position):
        execInfo = position.getExitOrder().getExecutionInfo()
        print "%s: SELL at $%.2f" % (execInfo.getDateTime(), execInfo.getPrice())
        self.__position = None
    def onExitCanceled(self, position):
        # If the exit was canceled, re-submit it.
        self.__position.exit()
    def onStart(self):
        print "Initial portfolio value: $%.2f" % self.getBroker().getEquity()
    def onFinish(self, bars):
        print "Final portfolio value: $%.2f" % self.getBroker().getEquity()
def main():
    feed = yahoofeed.Feed()
    feed.addBarsFromCSV("orcl", os.path.join(local_path,"orcl-2000.csv"))
    stg = ExtBrokeUpStrategy(feed, "orcl")
    stg.run()

if __name__ == '__main__':
    main()

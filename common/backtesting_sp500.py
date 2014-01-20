#!/usr/bin/env python
# author hongbin0908@126.com
import sys,os
local_path = os.path.dirname(os.path.abspath(sys.argv[0]))
sys.path.append(local_path + "./")
sys.path.append(local_path + "../common")

from pyalgotrade import strategy
from pyalgotrade.barfeed import yahoofeed
from pyalgotrade.strategy.position import LongPosition
from utils import *

class CommonStrategy(strategy.BacktestingStrategy):
    """
    a framework of a strategy
    """
    def __init__(self):
        feed = yahoofeed.Feed()
        instruments = get_sp500()
        for symbol in instruments:
            feed.addBarsFromCSV(symbol, os.path.join(get_stock_data_path(), symbol + ".csv"))
        strategy.BacktestingStrategy.__init__(self, feed,9999999999999)
        self.__positions = {}
        self.__instruments = instruments;
        for instrument in instruments:
            self.__positions[instrument] = []
    def display(self):
        for instrument in sorted(self.__positions.keys()):
            for position in self.__positions[instrument]:
                str = position.getInstrument()
                gains = 0
                if isinstance(position, LongPosition):
                    execInfo = position.getEntryOrder().getExecutionInfo()
                    buy_price = execInfo.getPrice()
                    str += "%s:BUY at %.2f " % (execInfo.getDateTime(), execInfo.getPrice())
                    execInfo = position.getExitOrder().getExecutionInfo()
                    shell_price = execInfo.getPrice()
                    str += "%s %s:SELL % %.2f " % (execInfo.getDateTime(), exeInfo.getPrice())
                    gains += (shell_price - buy_price) * position.getQuantity()
                    str += "gains:%.2f" % gains
                else:
                    assert(false)
    def summary(self):
        print ""

    def getPositions(self):
        return self.__positions
    def isLong(self, bar_ds):
        """
        must be override
        """
        assert(False)
    def isShort(self, bar_ds):
        """
        must be override
        """
        assert(False)
    def isSelltoCover(self, bar_ds, position):
        """
        must be override
        """
        assert(False)
    def isBuytoCover(self, bar_ds, position):
        """
        must be override
        """
        assert(False)
    def onEnterCanceled(self, position):
        self.__positions[position.getInstrument()][-1] = None
    def onExitCanceled(self, position):
        # If the exit was canceled, re-submit it.
        self.__positions[position.getInstrument()][-1].exit()

    def onBars(self, bars):
        for instrument in self.__instruments:
            bar = bars[instrument]
            if len(self.__positions[instrument]) == 0 or self.__positions[instrument][-1].exitFilled():
                if self.isLong(self.getFeed().getDataSeries(instrument)):
                    cur_cash = self.getBroker().getCash()
                    if (cur_cash > 10000):
                        cur_cash = 10000
                    quantity = cur_cash / bar.getClose()
                    self.__positions[instrument].append(self.enterLong(instrument, quantity ,True))
                if self.isShort(self.getFeed().getDataSeries(instrument)):
                    cur_cash = self.getBroker().getCash()
                    if (cur_cash > 10000):
                        cur_cash = 10000
                    quantity = cur_cash / bar.getClose()
                    self.__positions[instrument].append(self.enterShort(instrument, quantity ,True))
            else:
                if isinstance(self.__positions[instrument][-1], LongPosition):
                    if self.isSelltoCover(self.getFeed().getDataSeries(instrument), self.__positions[instrument][-1]):
                        self.__positions[instrument][-1].exit()


###########
# unit test of CommonStrategy
import unittest
class HoldOnStrategy(CommonStrategy):
    def isLong(self, bar_ds):
        return True
    def isShort(self, bar_ds):
        return False
    def isSelltoCover(self, bar_ds, position):
        return False
class CommonStrategyTest(unittest.TestCase):
    def setUp(self):
        self.__strategy = HoldOnStrategy()
    def test_1(self):
        self.__strategy.run()
        self.__strategy.display()
if __name__ == '__main__':
    unittest.main()

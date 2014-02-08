#!/usr/bin/env python
# author hongbin0908@126.com
# a common strategy 


import sys, os, time, re
local_path = os.path.dirname(os.path.abspath(sys.argv[0]))
sys.path.append(local_path + "/./")

import numpy 
import pyalgotrade
from pyalgotrade import strategy
from pyalgotrade.barfeed import yahoofeed
from pyalgotrade.tools import yahoofinance
from pyalgotrade import plotter

from utils import *


now = int(time.time())

# static
class UltraPosition:
    def __init__(self, position):
        """
        """
        self.__position = position
        self.__is_need_second_trade = False
        self.__pinfo = []
        self.__pending_exit_order = None
    def  isOpen(self):
        """ Returns True if the position is open.
        """
        return self.__position.isOpen()
        assert False
    def is_entered(self):
        """Returns True if the entry position entered
        """
        return self.__position.getEntryOrder().isFilled()
    def getCurExitOrder(self):
        return self.__pending_exit_order
    def get_entry_price(self):
        exe = self.__position.getEntryOrder().getExecutionInfo()
    def get_entry_time(self):
        return self.__position.getEntryOrder().getExecutionInfo().getDateTime()
    def get_entry_price(self):
        return self.__position.getEntryOrder().getExecutionInfo().getPrice()
        return exe.getPrice()
    def append_info(self, info):
        self.__pinfo.append(info)
class EntryInfo:
    def __init__(self):
        self.entryType = 0
        self.quantity = -1
        self.limitPrice = None
        self.stopPrice = None

class ExitInfo:
    def __init__(self):
        self.exitType = 0
        self.quantity = -1
        self.limitPrice = None
        self.stopPrice = None

class Multibacktesting(strategy.BacktestingStrategy):
    """
    TODO Comment
    __positions: a dict. key is instrument, the value is UltraPosition queue each is 
                 a position traded on this instrument
    """
    def __init__(self):
        #instruments = get_sp500_2()
        instruments = ['A', 'AA']
        tmp_dir = "/home/work/workplace/tmp"
        if not os.path.exists(tmp_dir):
            os.mkdir(tmp_dir)
        feed = yahoofinance.build_feed(instruments, 2010,2014,"/home/work/workplace/tmp/")
        strategy.BacktestingStrategy.__init__(self, feed)
        self.__instruments = instruments;
        self.__positions = {}
        for instrument in instruments:
            self.__positions[instrument] = []
    def long(self, instrument, datetime, barDs):
        assert False
    def short(self, instrument, datetime, barDs):
        assert False
    def exit(self, instrument, position, datetime, barDs):
        assert False
    def onBars(self, bars):
        for instrument in self.__instruments:
            barDs = self.getFeed().getDataSeries(instrument)
            bar = bars[instrument]
            datetime =  bar.getDateTime()
            dtstamp = time.mktime(datetime.timetuple())
            poss = self.__positions[instrument]
            if len(poss) == 0 or not poss[-1].isOpen():
                entry = self.long(instrument, datetime, barDs)
                if dtstamp > now - 24 * 7 * 3600 :
                    print "SCAN: long %s %s" % (datetime, instrument)
                if entry.quantity < 0:
                    entry.quantity = 10000
                if entry.entryType == 1:
                    pos = pyalgotrade.strategy.position.LongPosition(self,instrument, entry.limitPrice, entry.stopPrice, entry.quantity, False)
                    pos = UltraPosition(pos)
                    poss.append(pos)
            if len(poss) == 0 or not poss[-1].isOpen():
                entry = self.short(instrument, datetime, barDs)
                if dtstamp >  now - 24 * 7 * 3600:
                    print "SCAN sell %s %s" % (datetime, instrument)
                if entry.entryType == 2:
                    pos = pyalgotrade.strategy.position.ShortPosition(self, instrument, entry.limitPrice, entry.stopPrice, enry.quantity, False)
                    pos = UltraPosition(pos)
                    poss.append(pos)
            if len(poss) != 0 and poss[-1].is_entered():
                self.exit(instrument, poss[-1], datetime, barDs)
    def onEnterOk(self, position):
        instrument = position.getInstrument()
        pos = self.__positions[instrument]
        einfo = position.getEntryOrder().getExecutionInfo()
        #pos.append_info("%s entry success at %f"%( einfo.getDateTime(), einfo.getPrice()))
    def onExitOk(self, position):
        instrument = position.getInstrument()
        pos = self.__positions[instrument]
        entry = position.getEntryOrder().getExecutionInfo()
        exit = position.getExitOrder().getExecutionInfo()
        #print "%s trade: buy:%f %s sell:%f %s" % ( instrument, entry.getPrice(), entry.getDateTime(), exit.getPrice(), exit.getDateTime()),
        #gains = exit.getPrice()  - entry.getPrice()
        #gains = gains/entry.getPrice() * 100
        #print "gains: %.2f%%" % gains 
        #for info in self.__extinfos[instrument]["pos_info"]:
        #    print "\t\t", info
        #self.clean(instrument)
    def run_plot(self):
        plt = plotter.StrategyPlotter(self, False, True, True)
        self.run()
        plt.plot()


            



            

        
        
        

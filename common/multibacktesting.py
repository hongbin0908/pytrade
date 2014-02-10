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
    def exit(self, limitPrice=None, stopPrice=None, goodTillCanceled=None):
        self.__position.exit(limitPrice, stopPrice, goodTillCanceled)
    def get_infos(self):
        return self.__pinfo
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
                    # set printable infomation
                    pos.append_info("long position %s %d %s %s" % (datetime, entry.quantity, str(entry.limitPrice), str(entry.stopPrice)))
            if len(poss) == 0 or not poss[-1].isOpen():
                entry = self.short(instrument, datetime, barDs)
                if dtstamp >  now - 24 * 7 * 3600:
                    print "SCAN sell %s %s" % (datetime, instrument)
                if entry.entryType == 2:
                    pos = pyalgotrade.strategy.position.ShortPosition(self, instrument, entry.limitPrice, entry.stopPrice, enry.quantity, False)
                    pos = UltraPosition(pos)
                    poss.append(pos)
                    # set printable information
                    pos.append_info("short position %s %d %s %s" % (datetime, entry.quantity, str(entry.limitPrice), str(entry.stopPrice)))
            if len(poss) != 0 and poss[-1].is_entered() and poss[-1].isOpen():
                exit = self.exit(instrument, poss[-1], datetime, barDs)
                if exit.exitType == 1:
                    poss[-1].exit(exit.limitPrice, exit.stopPrice)
                    poss[-1].append_info("exit position %s %d %s %s" % (datetime, exit.quantity, str(exit.limitPrice), str(exit.stopPrice)))
                
    def onEnterOk(self, position):
        instrument = position.getInstrument()
        poss = self.__positions[instrument]
        einfo = position.getEntryOrder().getExecutionInfo()
        poss[-1].append_info("%s entry success at %f"%( einfo.getDateTime(), einfo.getPrice()))
    def onExitOk(self, position):
        instrument = position.getInstrument()
        poss = self.__positions[instrument]
        entry = position.getEntryOrder().getExecutionInfo()
        exit = position.getExitOrder().getExecutionInfo()
        poss[-1].append_info(" sell:%s %f" % (exit.getDateTime(), exit.getPrice()))
    def print_detail(self):
        for instrument in self.__positions:
            poss = self.__positions[instrument]
            print instrument,
            index = 1
            for pos in poss:
                print "\t#%d:", index
                for info in pos.get_infos():
                    print "\t\t%s", info
                index += 1


    def run_plot(self):
        plt = plotter.StrategyPlotter(self, False, True, True)
        self.run()
        self.print_detail()
        plt.plot()


            



            

        
        
        

#!/usr/bin/env python2.7
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
        #instruments = ['A', 'AA', 'AAPL', 'ABC', 'ABT', 'ACE', 'ACN', 'ACT', 'ADBE', 'ADI', 'ADM', 'ADP', 'ADS', 'ADSK', 'AEE', 'AEP', 'AES', 'AET', 'AFL', 'AGN', 'AIG', 'AIV', 'AIZ', 'AKAM', 'ALL', 'ALTR', 'ALXN', 'AMAT', 'AME', 'AMGN', 'AMP', 'AMT', 'AMZN', 'AN', 'AON', 'APA', 'APC', 'APD', 'APH', 'ARG', 'ATI', 'AVB', 'AVP', 'AVY', 'AXP', 'AZO', 'BA', 'BAC', 'BAX', 'BBBY', 'BBT', 'BBY', 'BCR', 'BDX', 'BEAM', 'BEN', 'BF-B', 'BHI', 'BIIB', 'BK', 'BLK', 'BLL', 'BMS', 'BMY', 'BRCM', 'BRK-B', 'BSX', 'BTU',  'BXP', 'C', 'CA', 'CAG', 'CAH', 'CAM', 'CAT', 'CB', 'CBG', 'CBS', 'CCE', 'CCI', 'CCL', 'CELG', 'CERN', 'CF', 'CFN', 'CHK', 'CHRW', 'CI', 'CINF', 'CL', 'CLF', 'CLX', 'CMA', 'CMCSA', 'CME', 'CMG', 'CMI', 'CMS', 'CNP', 'CNX', 'COF', 'COG', 'COH', 'COL', 'COP', 'COST', 'COV', 'CPB', 'CRM', 'CSC', 'CSCO', 'CSX', 'CTAS', 'CTL', 'CTSH', 'CTXS', 'CVC', 'CVS', 'CVX', 'D', 'DAL', 'DD', 'DE', 'DFS', 'DG', 'DGX', 'DHI', 'DHR', 'DIS', 'DISCA',  'DLTR', 'DNB', 'DNR', 'DO', 'DOV', 'DOW', 'DPS', 'DRI', 'DTE', 'DTV', 'DUK', 'DVA', 'DVN', 'EA', 'EBAY', 'ECL', 'ED', 'EFX', 'EIX', 'EL', 'EMC', 'EMN', 'EMR', 'EOG', 'EQR', 'EQT', 'ESRX', 'ESV', 'ETFC', 'ETN', 'ETR', 'EW', 'EXC', 'EXPD', 'EXPE', 'F', 'FAST', 'FCX', 'FDO', 'FDX', 'FE', 'FFIV', 'FIS', 'FISV', 'FITB', 'FLIR', 'FLR', 'FLS', 'FMC', 'FOSL', 'FOXA', 'FRX', 'FSLR', 'FTI', 'FTR', 'GAS', 'GCI', 'GD', 'GE',  'GHC', 'GILD', 'GIS', 'GLW',  'GME', 'GNW', 'GOOG', 'GPC', 'GPS', 'GRMN', 'GS', 'GT', 'GWW', 'HAL', 'HAR', 'HAS', 'HBAN', 'HCBK', 'HCN', 'HCP', 'HD', 'HES', 'HIG', 'HOG', 'HON', 'HOT', 'HP', 'HPQ', 'HRB', 'HRL', 'HRS', 'HSP', 'HST', 'HSY', 'HUM', 'IBM', 'ICE', 'IFF', 'IGT', 'INTC', 'INTU', 'IP', 'IPG', 'IR', 'IRM', 'ISRG', 'ITW', 'IVZ', 'JBL', 'JCI', 'JEC', 'JNJ', 'JNPR', 'JOY', 'JPM', 'JWN', 'K', 'KEY',  'TWX', 'TXN', 'TXT', 'TYC', 'UNH', 'UNM', 'UNP', 'UPS', 'URBN', 'USB', 'UTX', 'V', 'VAR', 'VFC', 'VIAB', 'VLO', 'VMC', 'VNO', 'VRSN', 'VRTX', 'VTR', 'VZ', 'WAG', 'WAT', 'WDC', 'WEC', 'WFC', 'WFM', 'WHR', 'WIN', 'WLP', 'WM', 'WMB', 'WMT', 'WU', 'WY', 'WYN', 'WYNN', 'X', 'XEL', 'XL', 'XLNX', 'XOM', 'XRAY', 'XRX', 'YHOO', 'YUM', 'ZION', 'ZMH']
        instruments = ['A', 'AA']
        feed = yahoofinance.build_feed(instruments, 2010,2014,"/tmp")
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
            poss = self.__positions[instrument]
            if len(poss) == 0 or not poss[-1].isOpen():
                entry = self.long(instrument, datetime, barDs)
                if entry.quantity < 0:
                    entry.quantity = 10000
                if entry.entryType == 1:
                    pos = pyalgotrade.strategy.position.LongPosition(self,instrument, entry.limitPrice, entry.stopPrice, entry.quantity, False)
                    pos = UltraPosition(pos)
                    poss.append(pos)
            if len(poss) == 0 or not poss[-1].isOpen():
                entry = self.short(instrument, datetime, barDs)
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


            



            

        
        
        

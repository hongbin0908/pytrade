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


class ExtBrokeUpStrategy(strategy.BacktestingStrategy):
    """
    __instrument : the code trading
    """
    def __init__(self, feed, instruments):
        strategy.BacktestingStrategy.__init__(self, feed)
        self.__instruments = instruments
        self.__positions = {}
        self.__extinfos = {} # store some useful information per instrment
        self.__stops= {}
        for instrument in instruments:
            self.clean(instrument)
    def try_entry(self, instrument, datetime, barDs):
        extinfo = self.__extinfos[instrument]
        pos_info = extinfo["pos_info"]
        #adx = pyalgotrade.talibext.indicator.ADX(barDs, 100,14)
        highs = barDs.getHighDataSeries()
        closes = barDs.getCloseDataSeries()
        opens = barDs.getOpenDataSeries()
        lows = barDs.getLowDataSeries()
        if len(highs)  < 240:
            return False
        cur_range = closes[-1] - opens[-1]
        for i in range(30):
            price_range = highs[-1 -i -1] - lows[-1 -i -1]
            if (cur_range < price_range):
                return False
        for i in range(239):
            if (highs[-1] < highs[-1 - i - 1]):
                return False
        pos_info.append("%s Entry Signal high:%f range:%f" % (datetime, highs[-1], highs[-1]-lows[-1]))
        return True
    def try_exit(self,instrument, datetime, barDs):
        position = self.__positions[instrument]
        highs = barDs.getHighDataSeries()
        closes = barDs.getCloseDataSeries()
        lows = closes = barDs.getLowDataSeries()
        cur_high = highs[-1]
        cur_close = closes[-1]
        cur_low = lows[-1]
        entry_price = position.getEntryOrder().getExecutionInfo().getPrice()
        if cur_close > entry_price * 1.10:
            if self.__stops[instrument] < cur_close * 0.95:
                self.__stops[instrument] = cur_close * 0.95
                position.cancelExit()
                position.exit(stopPrice = self.__stops[instrument])
                self.__extinfos[instrument]["pos_info"].append("%s Hit the target! set stop at %f" % (datetime, cur_close * 0.98))
        if (self.__stops[instrument] < 0) :
            self.__stops[instrument] = entry_price * 0.95
            position.cancelExit()
            position.exit(stopPrice = self.__stops[instrument])
            self.__extinfos[instrument]["pos_info"].append("%s set stop at %f" % (datetime, self.__stops[instrument]))
        else:
            entime = position.getEntryOrder().getExecutionInfo().getDateTime()
            enprice = position.getEntryOrder().getExecutionInfo().getPrice()
            delta = time.mktime(datetime.timetuple()) - time.mktime(entime.timetuple()) 
            if cur_high < enprice * 0.98  and  delta > 10 * 24 * 3600:
                position.cancelExit()
                position.exit()
                self.__extinfos[instrument]["pos_info"].append("%s close of time" % datetime)
            elif self.__stops[instrument] < cur_close * 0.95 :
                self.__stops[instrument] = cur_close * 0.95
                position.cancelExit()
                position.exit(stopPrice = self.__stops[instrument])
                self.__extinfos[instrument]["pos_info"].append("%s set stop at %f" % (datetime, self.__stops[instrument]))
        return False

    def clean(self, instrument):
        self.__stops[instrument] = -1
        self.__positions[instrument] = None
        self.__extinfos[instrument] = {}
        self.__extinfos[instrument]["pos_info"] = []

    def onBars(self, bars):
        for instrument in self.__instruments:
            extinfo = self.__extinfos[instrument]
            barDs = self.getFeed().getDataSeries(instrument)
            bar = bars[instrument]
            datetime =  bar.getDateTime()
            if self.__positions[instrument] == None:
                if self.try_entry(instrument, datetime, barDs):
                    stop_price = bar.getHigh() * 1.02 
                    quantity = 10000.0/stop_price
                    self.__positions[instrument] = self.enterLongStop(instrument, stop_price, quantity, True)
                    extinfo["pos_info"].append("%s set a stop order at %f" % (datetime, stop_price))
                    self.__extinfos[instrument]["entry_time"] = datetime;
            elif not self.__positions[instrument].entryFilled():
                cur_timestamp = time.mktime(datetime.timetuple())
                entry_timestamp = time.mktime(self.__extinfos[instrument]["entry_time"].timetuple())
                if cur_timestamp > entry_timestamp + 5:
                    self.__positions[instrument].cancelEntry()
                    self.clean(instrument)
            else:
                self.try_exit(instrument, datetime, barDs)
    def onEnterOk(self, position):
        instrument = position.getInstrument()
        extinfo = self.__extinfos[instrument]
        einfo = position.getEntryOrder().getExecutionInfo()
        extinfo["pos_info"].append("%s entry success at %f"%( einfo.getDateTime(), einfo.getPrice()))

    def onExitOk(self, position):
        instrument = position.getInstrument()
        position = self.__positions[instrument]
        entry = position.getEntryOrder().getExecutionInfo()
        exit = position.getExitOrder().getExecutionInfo()
        print "%s trade: buy:%f %s sell:%f %s" % ( instrument, entry.getPrice(), entry.getDateTime(), exit.getPrice(), exit.getDateTime()),
        gains = exit.getPrice()  - entry.getPrice()
        gains = gains/entry.getPrice() * 100
        print "gains: %.2f%%" % gains 
        for info in self.__extinfos[instrument]["pos_info"]:
            print "\t\t", info
        self.clean(instrument)

def main():
    instruments = ['A', 'AA', 'AAPL', 'ABC', 'ABT', 'ACE', 'ACN', 'ACT', 'ADBE', 'ADI', 'ADM', 'ADP', 'ADS', 'ADSK', 'AEE', 'AEP', 'AES', 'AET', 'AFL', 'AGN', 'AIG', 'AIV', 'AIZ', 'AKAM', 'ALL', 'ALTR', 'ALXN', 'AMAT', 'AME', 'AMGN', 'AMP', 'AMT', 'AMZN', 'AN', 'AON', 'APA', 'APC', 'APD', 'APH', 'ARG', 'ATI', 'AVB', 'AVP', 'AVY', 'AXP', 'AZO', 'BA', 'BAC', 'BAX', 'BBBY', 'BBT', 'BBY', 'BCR', 'BDX', 'BEAM', 'BEN', 'BF-B', 'BHI', 'BIIB', 'BK', 'BLK', 'BLL', 'BMS', 'BMY', 'BRCM', 'BRK-B', 'BSX', 'BTU',  'BXP', 'C', 'CA', 'CAG', 'CAH', 'CAM', 'CAT', 'CB', 'CBG', 'CBS', 'CCE', 'CCI', 'CCL', 'CELG', 'CERN', 'CF', 'CFN', 'CHK', 'CHRW', 'CI', 'CINF', 'CL', 'CLF', 'CLX', 'CMA', 'CMCSA', 'CME', 'CMG', 'CMI', 'CMS', 'CNP', 'CNX', 'COF', 'COG', 'COH', 'COL', 'COP', 'COST', 'COV', 'CPB', 'CRM', 'CSC', 'CSCO', 'CSX', 'CTAS', 'CTL', 'CTSH', 'CTXS', 'CVC', 'CVS', 'CVX', 'D', 'DAL', 'DD', 'DE', 'DFS', 'DG', 'DGX', 'DHI', 'DHR', 'DIS', 'DISCA',  'DLTR', 'DNB', 'DNR', 'DO', 'DOV', 'DOW', 'DPS', 'DRI', 'DTE', 'DTV', 'DUK', 'DVA', 'DVN', 'EA', 'EBAY', 'ECL', 'ED', 'EFX', 'EIX', 'EL', 'EMC', 'EMN', 'EMR', 'EOG', 'EQR', 'EQT', 'ESRX', 'ESV', 'ETFC', 'ETN', 'ETR', 'EW', 'EXC', 'EXPD', 'EXPE', 'F', 'FAST', 'FCX', 'FDO', 'FDX', 'FE', 'FFIV', 'FIS', 'FISV', 'FITB', 'FLIR', 'FLR', 'FLS', 'FMC', 'FOSL', 'FOXA', 'FRX', 'FSLR', 'FTI', 'FTR', 'GAS', 'GCI', 'GD', 'GE',  'GHC', 'GILD', 'GIS', 'GLW',  'GME', 'GNW', 'GOOG', 'GPC', 'GPS', 'GRMN', 'GS', 'GT', 'GWW', 'HAL', 'HAR', 'HAS', 'HBAN', 'HCBK', 'HCN', 'HCP', 'HD', 'HES', 'HIG', 'HOG', 'HON', 'HOT', 'HP', 'HPQ', 'HRB', 'HRL', 'HRS', 'HSP', 'HST', 'HSY', 'HUM', 'IBM', 'ICE', 'IFF', 'IGT', 'INTC', 'INTU', 'IP', 'IPG', 'IR', 'IRM', 'ISRG', 'ITW', 'IVZ', 'JBL', 'JCI', 'JEC', 'JNJ', 'JNPR', 'JOY', 'JPM', 'JWN', 'K', 'KEY',  'TWX', 'TXN', 'TXT', 'TYC', 'UNH', 'UNM', 'UNP', 'UPS', 'URBN', 'USB', 'UTX', 'V', 'VAR', 'VFC', 'VIAB', 'VLO', 'VMC', 'VNO', 'VRSN', 'VRTX', 'VTR', 'VZ', 'WAG', 'WAT', 'WDC', 'WEC', 'WFC', 'WFM', 'WHR', 'WIN', 'WLP', 'WM', 'WMB', 'WMT', 'WU', 'WY', 'WYN', 'WYNN', 'X', 'XEL', 'XL', 'XLNX', 'XOM', 'XRAY', 'XRX', 'YHOO', 'YUM', 'ZION', 'ZMH']
    feed = yahoofinance.build_feed(instruments, 2010,2014,"/tmp")
    stg = ExtBrokeUpStrategy(feed, instruments)
    plt = plotter.StrategyPlotter(stg, False, True, True)
    stg.run()
    plt.plot()


if __name__ == '__main__':
    main()

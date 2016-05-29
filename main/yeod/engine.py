#!/usr/bin/env python2.7

import os, sys
from datetime import date
import time
from pyalgotrade.tools import yahoofinance
import multiprocessing
import finsymbols

def _single(symbol, data_dir): 
    retry = 3
    eod = None
    while retry > 0:
        try:
            eod = yahoofinance.download_csv(symbol, date(1970,01,01),date(2099,01,01), 'd')
            with open(os.path.join(data_dir,symbol+".csv"), 'w') as fout:
                print >> fout, eod
        except Exception,ex:
            #if int(ex.getcode()) == 404:
            #    print symbol, "404, just break"
            #    break
            print symbol, Exception, ":", ex.getcode(), " ", ex
            time.sleep(61)
            retry -=1
            continue
        break   	
    if not eod is None:
        return len(eod)
    return -1

def work(syms,data_dir, processes):
    syms.sort()
    pool = multiprocessing.Pool(processes = int(processes) )
    result = {}
    for sym in syms:
        if sym.find('^') > 0:
            continue
        if sym.find('.') > 0:
            continue
        result[sym] = pool.apply_async(_single, (sym, data_dir))
    print len(result)
    for sym in result:
        print sym, result[sym].get()

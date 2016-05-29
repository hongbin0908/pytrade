#!/usr/bin/env python2.7

import os, sys
from datetime import date
from pyalgotrade.tools import yahoofinance
import multiprocessing
import finsymbols

def _single(symbol, data_dir): 
    retry = 3
    while retry > 0:
        try:
            eod = yahoofinance.download_csv(symbol, date(1970,01,01),date(2099,01,01), 'd')
            with open(os.path.join(data_dir,symbol+".csv"), 'w') as fout:
                print >> fout, eod
        except Exception,ex:
            print symbol, Exception, ":", ex
            retry -=1
            continue
        break
    return len(eod)

def work(syms,data_dir, processes):
    pool = multiprocessing.Pool(processes = int(processes) )
    result = {}
    for sym in syms:
        result[sym] = pool.apply_async(_single, (sym, data_dir))
    for sym in syms:
        print sym, result[sym].get()

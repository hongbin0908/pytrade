#!/usr/bin/env python2.7

import os, sys
from datetime import date
import time
from pyalgotrade.tools import yahoofinance
from urllib2 import  HTTPError
import multiprocessing
import finsymbols
local_path = os.path.dirname(__file__)

def _single(symbol, data_dir): 
    retry = 3
    eod = None
    while retry > 0:
        try:
            eod = yahoofinance.download_csv(symbol, date(1970,01,01),date(2099,01,01), 'd')
            with open(os.path.join(data_dir,symbol+".csv"), 'w') as fout:
                print >> fout, eod
        except Exception,ex:
            if isinstance(ex, HTTPError) and int(ex.getcode()) == 404:
                print symbol, "404, just break"
                break
            print symbol, Exception, ":", ex.getcode(), " ", ex
            time.sleep(61)
            retry -=1
            continue
        break   	
    if not eod is None:
        return len(eod)
    return -1

def load_blacklist():
    d = set([])
    with open(os.path.join(local_path, 'blacklist')) as f:
        for each in f.readlines():
            d.add(each.strip())
    return d
def work(syms,data_dir, processes):
    blacklist = load_blacklist()
    syms.sort()
    pool = multiprocessing.Pool(processes = int(processes) )
    result = {}
    for sym in syms:
        if sym.find('^') > 0:
            continue
        if sym.find('.') > 0:
            continue
        if os.path.isfile(os.path.join(data_dir, sym + ".csv")):
            continue
        if sym in blacklist:
            continue
        result[sym] = pool.apply_async(_single, (sym, data_dir))
    print len(result)
    pool.close()
    pool.join()
    succ = 0; fail = 0
    for each in result:
        if result[each] > 0: succ += 1
        else: fail += 1
    return succ, fail

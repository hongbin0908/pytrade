#!/usr/bin/env python
import os, sys, json, MySQLdb, datetime

local_path = os.path.dirname(os.path.abspath(sys.argv[0]))
sys.path.append(local_path + "/./")
sys.path.append(local_path + "/../yahoo-finance/")
print sys.path
from yahoo_finance import Share
from yahoo_finance.mult import MShare

def get_stock(s_date, symbol):
    stock = Share(symbol)
    retry = 3
    lStock = []
    while retry >= 0:
        try:
            lStock = stock.get_historical(s_date, s_date)
        except Exception, ex:
            retry -= 1
            continue
        break
    if len(lStock) == 0:
        return None
    return lStock[0]

def get_stock_range(s_start, s_end, symbol):
    stock = Share(symbol)
    retry = 3
    lStock = []
    while retry >= 0:
        try:
            lStock = stock.get_historical(s_start, s_end)
        except Exception, ex:
            retry -= 1
            continue
        isvalid = True
        for dStock in lStock:
            if not 'Open' in dStock:
                isvalid = False
                break
        if isvalid:
            break
    if len(lStock) == 0:
        return None
    return lStock

def get_stock_range_mult(s_start, s_end, lSymbol):
    stock = MShare(lSymbol)
    retry = 3
    lStock = []
    while retry >= 0:
        try:
            lStock = stock.get_historical(s_start, s_end)
        except Exception, ex:
            retry -= 1
            continue
        isvalid = True
        for dStock in lStock:
            if not 'Open' in dStock:
                isvalid = False
                break
        if isvalid:
            break
    if len(lStock) == 0:
        return None
    return lStock

if __name__ == '__main__':
    print get_stock('2014-01-17', 'YHOO')
    

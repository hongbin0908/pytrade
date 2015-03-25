#!/usr/bin/env python
import os, sys
import json, MySQLdb, datetime

local_path = os.path.dirname(os.path.abspath(sys.argv[0]))
sys.path.append(local_path + "/./")

from stock_finance_helper import *
from stock_db_helper import *

def get_batch(s_start, s_now, sym_stack, ttl = 3):
    if len(sym_stack) == 0:
        return
    if ttl <= 0:
        return 
    ld_stock_data = get_stock_range_mult(s_start, s_now, sym_stack)
    if ld_stock_data == None:
        return 
    c_sym_stack = set(sym_stack)
    c_sym_stack2 = set()
    for stock_data in ld_stock_data:
        sym = stock_data["Symbol"]
        c_sym_stack2.add(sym)
        date =  stock_data["Date"]
        volume = int(stock_data["Volume"])
        adj_close = float(stock_data["Adj_Close"])
        high = float(stock_data["High"])
        low = float(stock_data["Low"])
        close_price = float(stock_data["Close"])
        open_price = float(stock_data["Open"])
        insert_stock_data_daily(sym, date, volume, open_price, close_price, high, low, adj_close)
    if len(c_sym_stack2) != 0:
        get_batch(s_start, s_now, c_sym_stack - c_sym_stack2)
    else:
        get_batch(s_start, s_now, c_sym_stack, ttl - 1)
if __name__ == '__main__':
    d_now = datetime.datetime.now()
    for i in xrange(0,10):
        d_start = d_now - datetime.timedelta(days=i)
        s_start = d_start.strftime('%Y-%m-%d')
        sym_stack = []
        sym_stack_num = 0
        print len(get_sp500_list())
        for symbol in  get_sp500_list():
            if len(sym_stack) < 1000:
                sym_stack.append(symbol)
                continue
            if check_is_share_ready(sym_stack, s_start):
                continue
            get_batch(s_start, s_start, sym_stack)
            sym_stack = []
            sym_stack.append(symbol)

        if not check_is_share_ready(sym_stack, s_start):
            get_batch(s_start, s_start, sym_stack)

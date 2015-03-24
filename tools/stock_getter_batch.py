#!/usr/bin/env python
import os, sys
import json, MySQLdb, datetime

local_path = os.path.dirname(os.path.abspath(sys.argv[0]))
sys.path.append(local_path + "/./")

from stock_finance_helper import *
from stock_db_helper import *

def get_batch(s_start, s_now, sym_stack):
    ld_stock_data = get_stock_range_mult(s_start, s_now, sym_stack)
    if ld_stock_data == None:
        assert(False)
    for stock_data in ld_stock_data:
        sym = stock_data["Symbol"]
        date =  stock_data["Date"]
        volume = int(stock_data["Volume"])
        adj_close = float(stock_data["Adj_Close"])
        high = float(stock_data["High"])
        low = float(stock_data["Low"])
        close_price = float(stock_data["Close"])
        open_price = float(stock_data["Open"])
        insert_stock_data_daily(sym, date, volume, open_price, close_price, high, low, adj_close)

if __name__ == '__main__':
    d_now = datetime.datetime.now()
    s_now = d_now.strftime('%Y-%m-%d')
    s_start =  get_start_date()
    sym_stack = []
    sym_stack_num = 0
    for symbol in  get_sp500_list():
        print symbol
        if len(sym_stack) < 20:
            sym_stack.append(symbol)
            continue
        get_batch(s_start, s_now, sym_stack)
        sym_stack = []

    get_batch(s_start, s_now, sym_stack)


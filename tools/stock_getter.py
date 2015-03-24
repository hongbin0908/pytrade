#!/usr/bin/env python
import os, sys
import json, MySQLdb, datetime

local_path = os.path.dirname(os.path.abspath(sys.argv[0]))
sys.path.append(local_path + "/./")

import finsymbols
from yahoo_finance import Share
from stock_finance_helper import *
from stock_db_helper import *

if __name__ == '__main__':
    d_now = datetime.datetime.now()
    while True: 
        s_now = d_now.strftime('%Y-%m-%d')
        if s_now < get_start_date():
            break
        print s_now
        if is_valid_date(s_now):
            for symbol in  get_sp500_list():
                stock_data = get_stock(s_now, symbol)
                if stock_data == None:
                    continue
                {'Volume': '2216600', 'Symbol': 'A', 'Adj_Close': '42.21', 'High': '42.53', 'Low': '42.20', 'Date': '2015-03-20', 'Close': '42.21', 'Open': '42.39'}
                assert(symbol == stock_data["Symbol"])
                assert(s_now == stock_data["Date"])
                volume = stock_data["Volume"]
                adj_close = stock_data["Adj_Close"]
                high = stock_data["High"]
                low = stock_data["Low"]
                close_price = stock_data["Close"]
                open_price = stock_data["Open"]
                insert_stock_data_daily(symbol, s_now, volume, open_price, close_price, high, low, adj_close)

        d_now = d_now - datetime.timedelta(days=1)

#!/usr/bin/env python
import os, sys
import json, MySQLdb, datetime

local_path = os.path.dirname(os.path.abspath(sys.argv[0]))
sys.path.append(local_path + "/./")

import finsymbols
from yahoo_finance import Share

conn = None


def get_conn():
    global conn
    if conn == None:
        conn = MySQLdb.connect(host="123.56.128.198", user="root", passwd="123456", db="mytrade", charset="utf8")

def get_start_date():
    get_conn()
    cursor = conn.cursor()
    sql = "select datetime from exchange_days order by `datetime` asc limit 1"
    n = cursor.execute(sql)
    rows = cursor.fetchall()
    return rows[0][0]
    cursor.close()
def is_valid_date(s_date):
    result = False
    get_conn()
    cursor = conn.cursor()
    sql = "select isvalid from exchange_days where datetime=%s and isvalid=1"
    params = (s_date)
    n = cursor.execute(sql, params)
    if n == 1 :
        result = True
    cursor.close()
    return result
def get_sp500_list():
    get_conn()
    result = []
    cursor = conn.cursor()
    sql = "select symbol from symbols where stock_index = %s and isvalid = %s"
    params = ('sp500',1)
    n = cursor.execute(sql, params)
    for row in cursor.fetchall():
        result.append(row[0])
    cursor.close()
    return result
def query_stock_data_daily(symbol, date):
    result = dict({})
    get_conn()
    cursor = conn.cursor()
    sql = "select symbol, date, volume, open, close, high, low, adj_close from stock_data_daily where symbol = %s and date = %s"
    params = (symbol, date)
    n = cursor.execute(sql, params)
    if n != 0:
        assert(n == 1)
        rows = cursor.fetchall()
        if len(rows) > 0:
            assert (len(rows) == 1)
            row = rows[0]
            assert (len(row) == 8)
            result["symbol"] = row[0]
            result["date"] = row[1]
            result["volume"] = row[2]
            result["open"] = row[3]
            result["close"] = row[4]
            result["high"] = row[5]
            result["low"] = row[6]
            result["adj_close"] = row[7]
        else:
            result = None
    else:
        result = None

    cursor.close()
    return result

    return 
def insert_stock_data_daily(symbol, date, volume, open_price, close_price, high, low, adj_close):
    if query_stock_data_daily(symbol, date) != None:
        return False
    get_conn()
    cursor = conn.cursor()
    assert(volume > 0)
    assert(high >= close_price)
    print low 
    print close_price
    assert(low <= close_price)
    sql = "insert into stock_data_daily (symbol, date, volume, open, close, high, low, adj_close) values (%s, %s, %s,%s,%s,%s,%s,%s)"
    params = (symbol, date, volume, open_price, close_price, high, low, adj_close)
    n = cursor.execute(sql, params)
    assert (n == 1)
    conn.commit()
    cursor.close()
    return True


if __name__ == '__main__':
    #print is_valid_date("2014-01-13")
    #print get_sp500_list()
    print insert_stock_data_daily("TEST", "2014-01-15", 100, 1.0,1.0,1.0,1.0,1.0)

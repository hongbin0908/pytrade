#!/usr/bin/env python
import os, sys, json, MySQLdb

local_path = os.path.dirname(os.path.abspath(sys.argv[0]))
sys.path.append(local_path + "/./")

'''
load the sp500 into database
'''
import finsymbols


def get_sp500():#{{{
    return finsymbols.symbols.get_sp500_symbols()
#}}}


if __name__ == '__main__':
    conn = MySQLdb.connect(host="localhost", user="root", passwd="123456", db="mytrade", charset="utf8")
    cursor = conn.cursor()

    lSymbols =  get_sp500()
    for sym in lSymbols:
        sql = "insert into symbols(symbol,sector,industry,exchange,headquaters, company, stock_index, isvalid)" \
            +"values(%s,%s,%s,%s,%s,%s,%s,%s); "
        params = (sym["symbol"], sym["sector"], sym["industry"], "none", sym["headquaters"], sym["company"], "sp500", 1)
        n =cursor.execute(sql, params)
        print sql, params
    cursor.close() 
    conn.commit()
    conn.close()

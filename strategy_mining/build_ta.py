
#!/usr/bin/env python
# -*- coding: utf-8 -*-

#@author 
import os,sys
import talib
local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..')
sys.path.append(root)
import strategy_mining.model_base as base
import ta


def main():
    for each in base.get_file_list(os.path.join(local_path, '..', 'data', 'yeod')):
        symbol = base.get_stock_from_path(each)
        df = base.get_stock_data_pd(base.get_stock_from_path(each))
        df = ta.cal_all(df)
        df.to_csv(os.path.join(root, 'data', 'ta', symbol + ".csv"))
if __name__ == '__main__':
    main()



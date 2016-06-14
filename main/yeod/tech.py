#!/usr/bin/env python2.7

import os, sys
import finsymbols
local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..')
sys.path.append(root)
from yeod import  engine


def get_data_root():
    data_root = os.path.join(root, "data", 'yeod_tech')
    if not os.path.exists(data_root):
        os.makedirs(data_root)
    return data_root

def get():
    syms = set()
    for each in finsymbols.symbols.get_nasdaq_symbols():
        if each['symbol'] == 'GOOG':
            print each
        if each['industry'] == 'Technology' or each['sector'] == 'Technology':
            syms.add(each['symbol'].strip())
    for each in finsymbols.symbols.get_nyse_symbols():
        if each['industry'] == 'Technology' or each['sector'] == 'Technology':
            syms.add(each['symbol'].strip())
    return list(syms)

def main(argv):
    print engine.work(get(), get_data_root(), sys.argv[1])
if __name__ == '__main__':
    main(sys.argv)

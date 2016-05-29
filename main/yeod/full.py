#!/usr/bin/env python2.7

import os, sys
import finsymbols
local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..')
sys.path.append(root)
from yeod import  engine


def get_data_root():
    data_root = os.path.join(root, "data", 'yeod_full')
    if not os.path.exists(data_root):
        os.makedirs(data_root)
    return data_root

def get_full():
    syms = set()
    for each in finsymbols.symbols.get_nasdaq_symbols():
        syms.add(each['symbol'])
    for each in finsymbols.symbols.get_nyse_symbols():
        syms.add(each['symbol'])
    return list(syms)

def main(argv):
    engine.work(get_full(), get_data_root(), sys.argv[1])
if __name__ == '__main__':
    main(sys.argv)

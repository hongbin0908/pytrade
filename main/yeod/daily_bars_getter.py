
#!/usr/bin/env python2.7

import os, sys
import finsymbols
local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..')
sys.path.append(root)
from yeod import  engine

def get_data_root():
    data_root = os.path.join(root, "data", 'yeod')
    if not os.path.exists(data_root):
        os.makedirs(data_root)
    return data_root

def get_sp500():
    symbols = []
    for each in finsymbols.symbols.get_sp500_symbols():
        symbols.append(each['symbol'])
    return symbols  

def main(argv):
    engine.work(get_sp500(), get_data_root(), sys.argv[1])
if __name__ == '__main__':
    main(sys.argv)

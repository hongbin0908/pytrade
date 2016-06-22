
#!/usr/bin/env python2.7

import os, sys
import finsymbols
local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..')
sys.path.append(os.path.join(root,'..'))
from main.yeod import engine

def get_data_root():
    data_root = os.path.join(root, "data", 'dow')
    if not os.path.exists(data_root):
        os.makedirs(data_root)
    return data_root

def get():
    symbols = [ "AAPL", "AXP", "BA", "CAT", "CSCO", 
                "CVX", "DD", "DIS", "GE", "GS", "HD", 
                "IBM", "INTC", "JNJ", "JPM", "MCD", 
                "MMM", "MRK", "MSFT", "NKE", "PFE", 
                "TRV", "UNH", "UTX", "V", "VZ", "WMT", 
                "XOM", ]
    return symbols 

def main(argv):
    return engine.work(get(), get_data_root(), argv[1])
if __name__ == '__main__':
    main(sys.argv)

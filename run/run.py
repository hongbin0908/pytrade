#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

#@author  Bin Hong

import sys,os

local_path = os.path.dirname(__file__)
root = os.path.join(local_path,'..')
sys.path.append(root)


from main.yeod import yeod
from main.yeod_b import yeod_b
from main.ta import build
from main.ta import build_b
from main.utils import time_me
from main.pred import pred
from main.pred_b import pred_b
import main.base as base

@time_me
def main(argv):
    os.rmdir(os.path.join(root, "data", "yed_batch", "sp500Top100"))
    yeod.main(["index_dow", 1])
    yeod_b.main(["sp500Top100", 50, 10])
    build_b.main(["sp500Top100", 50 ,'call1s3',10])
    paper_b.main(["GBCv1n1000md3lr001-call1s3-sp500Top100-50-label5-1700-01-01-2009-12-31-0-0", 650, "call1s3-sp500Top100", 50, "2010-06-01", "2016-06-31", 2, 400])
    last_date = base.last_trade_date()
    pred_d.main(["GBCv1n1000md3lr001-call1s3-sp500Top100-50-label5-1700-01-01-2009-12-31-0-0", 650, "call1s3-sp500Top100-50",
    "2010-06-03",
    "2010-06-03", "label5"])
if __name__ == '__main__':
    main(sys.argv[1:])

#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

#@author  Bin Hong

import sys,os,shutil

local_path = os.path.dirname(__file__)
root = os.path.join(local_path,'..')
sys.path.append(root)

from main.yeod import yeod
from main.yeod import yeod_b
from main.ta import build
from main.ta import build_b
from main.utils import time_me
from main.pred import pred_b
from main.paper import paper_b
import main.base as base

ta = "call1s4"
eod = "sp500Top100"
batch=50
model = "GBCv1n1000md3lr001-%s-sp500Top100-%d-label5-1700-01-01-2009-12-31-0-0" % (ta, batch)
@time_me
def main(argv):
    #shutil.rmtree(os.path.join(root, "data", "yeod_batch", "%s-%d" % (eod, batch)), ignore_errors=False)
    yeod.main(["index_dow", 1])
    yeod_b.main([eod, batch, 10])
    build_b.main([eod, batch ,ta,10])
    paper_b.main([model, 600, "%s-%s" % (ta,eod), batch, "2010-06-01", "2016-06-31", 2, 400])
    last_date = base.last_trade_date()
    pred_b.main([model, 600, "%s-%s-%d" % (ta, eod, batch),
        last_date, last_date, "label5"])
if __name__ == '__main__':
    main(sys.argv[1:])


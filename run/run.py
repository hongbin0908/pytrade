#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

#@author  Bin Hong

import sys,os

local_path = os.path.dirname(__file__)
root = os.path.join(local_path,'..')
sys.path.append(root)


from main.yeod import yeod
from main.ta import build
from main.utils import time_me
from main.model import modeling as model
from main.pred import pred
import main.base as base

@time_me
def main(argv):
    yeod.main(["dow", 5])
    build.main(["dow", 'call1s1',5])
    last_date = base.last_trade_date()
    pred.main(['call1s1_dow_GBCv1n322md3lr001_l5_s1700e2009', 'call1s1_dow', last_date, last_date, 'label5'])
if __name__ == '__main__':
    main(sys.argv[1:])

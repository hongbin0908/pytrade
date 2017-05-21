#! /usr/bin/env python2.7
# -*- coding: utf-8 -*-
# @author  Bin Hong

import os
import sys
import logging
logging.basicConfig(level=logging.WARN)

local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..')
sys.path.append(root)

from main.yeod import yeod
from main import base

#yeod.main2(1, "/home/hongbin/misc/nginx/html/yeod/index.zip", yeod.index().get_syms())
last_trade_date = base.get_last_trade_date()
#yeod.main2(1, "/home/hongbin/misc/nginx/html/yeod/index_%s.zip" % last_trade_date , yeod.index().get_syms())
yeod.main2(2, '/tmp/sp500_%s' % last_trade_date, "/home/hongbin/misc/nginx/html/yeod/sp500_snapshot_20091231_%s.zip" % last_trade_date,  yeod.sp500_snapshot("sp500_snapshot_20091231").get_syms())
yeod.main2(2, '/tmp/sp500_%s' % last_trade_date, "/home/hongbin/misc/nginx/html/yeod/sp500_snapshot_20101207_%s.zip" % last_trade_date,  yeod.sp500_snapshot("sp500_snapshot_20101207").get_syms())
yeod.main2(10,'/tmp/sp500_%s' % last_trade_date,  "/home/hongbin/misc/nginx/html/yeod/sp500_snapshot_20111231_%s.zip" % last_trade_date, yeod.sp500_snapshot("sp500_snapshot_20111231").get_syms())
yeod.main2(10,'/tmp/sp500_%s' % last_trade_date,  "/home/hongbin/misc/nginx/html/yeod/sp500_snapshot_20121229_%s.zip" % last_trade_date, yeod.sp500_snapshot("sp500_snapshot_20121229").get_syms())
yeod.main2(10,'/tmp/sp500_%s' % last_trade_date,  "/home/hongbin/misc/nginx/html/yeod/sp500_snapshot_20131229_%s.zip" % last_trade_date, yeod.sp500_snapshot("sp500_snapshot_20131229").get_syms())
yeod.main2(10,'/tmp/sp500_%s' % last_trade_date,  "/home/hongbin/misc/nginx/html/yeod/sp500_snapshot_20141229_%s.zip" % last_trade_date, yeod.sp500_snapshot("sp500_snapshot_20141229").get_syms())
yeod.main2(10,'/tmp/sp500_%s' % last_trade_date,  "/home/hongbin/misc/nginx/html/yeod/sp500_snapshot_20151228_%s.zip" % last_trade_date, yeod.sp500_snapshot("sp500_snapshot_20151228").get_syms())

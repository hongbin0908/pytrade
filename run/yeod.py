#! /usr/bin/env python2.7
# -*- coding: utf-8 -*-
# @author  Bin Hong

import os
import sys
import time
import shutil
import subprocess
import datetime
import logging
logging.basicConfig(level=logging.WARN)

local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..')
sys.path.append(root)

from main.yeod import yeod

#yeod.main2(10, "/home/hongbin/misc/nginx/html/yeod/sp100_snapshot_20091129.zip",
#        yeod.sp100_snapshot("20091129").get_syms())
#yeod.main2(10, "/home/hongbin/misc/nginx/html/yeod/sp100_snapshot_20100710.zip",
#        yeod.sp100_snapshot("20100710").get_syms())
#yeod.main2(10, "/home/hongbin/misc/nginx/html/yeod/sp100_snapshot_20140321.zip",
#        yeod.sp100_snapshot("20140321").get_syms())
#yeod.main2(10, "/home/hongbin/misc/nginx/html/yeod/sp100_snapshot_20161110.zip",
#        yeod.sp100_snapshot("20161110").get_syms())
syms = yeod.sp500_snapshot("20091231").get_syms()
print(len(syms))
yeod.main2(10, "/home/hongbin/misc/nginx/html/yeod/sp500_snapshot_20091231.zip",
        yeod.sp500_snapshot("20091231").get_syms())
yeod.main2(1, "/home/hongbin/misc/nginx/html/yeod/index.zip",
        yeod.index.get_syms())

yeod.main2(10, "/home/hongbin/misc/nginx/html/yeod/sp500_snapshot_20091231.zip",
        yeod.sp500_snapshot("20091231").get_syms())
yeod.main2(10, "/home/hongbin/misc/nginx/html/yeod/sp500_snapshot_20101207.zip",
        yeod.sp500_snapshot("20101207").get_syms())
yeod.main2(10, "/home/hongbin/misc/nginx/html/yeod/sp500_snapshot_20111231.zip",
        yeod.sp500_snapshot("20111231").get_syms())
yeod.main2(10, "/home/hongbin/misc/nginx/html/yeod/sp500_snapshot_20121229.zip",
        yeod.sp500_snapshot("20121229").get_syms())
yeod.main2(10, "/home/hongbin/misc/nginx/html/yeod/sp500_snapshot_20131229.zip",
        yeod.sp500_snapshot("20131229").get_syms())
yeod.main2(10, "/home/hongbin/misc/nginx/html/yeod/sp500_snapshot_20141229.zip",
        yeod.sp500_snapshot("20141229").get_syms())
yeod.main2(10, "/home/hongbin/misc/nginx/html/yeod/sp500_snapshot_20151228.zip",
        yeod.sp500_snapshot("20151228").get_syms())
yeod.main2(1, "/home/hongbin/misc/nginx/html/yeod/index.zip", yeod.index().get_syms())

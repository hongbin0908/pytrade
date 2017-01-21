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
logging.basicConfig(level=logging.DEBUG)

local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..')
sys.path.append(root)

from main.yeod import yeod


def git_diff():
    s = subprocess.check_output("""
            cd {local_path};
            git diff | wc -l
    """.format(**{"local_path":local_path}), shell=True)
    return int(s)
dirname = "sp100_%s" % datetime.datetime.now().strftime("%Y%m%d")
    
dirpath = os.path.join(local_path, dirname)
if os.path.exists(dirpath):
    shutil.rmtree(dirpath)
yeod.main2(5, dirpath)
#yeod.main2(5, dirpath, ["GOOG", "YHOO"])

cmdstr = """
    pwd;
    cd {local_path}/{dirname};
    tar cvzf ../{dirname}.tar.gz *.csv;
    cd ..;
    mv {dirname}.tar.gz /home/hongbin/misc/nginx/html;
    rm -rf /home/hongbin/misc/nginx/html/sp100_current.tar.gz;
    cp -rf /home/hongbin/misc/nginx/html/{dirname}.tar.gz /home/hongbin/misc/nginx/html/sp100_current.tar.gz 
""".format(**{"dirname":dirname, "local_path": local_path})
print(cmdstr)
subprocess.check_output(cmdstr, shell=True)

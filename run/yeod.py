#! /usr/bin/env python2.7
# -*- coding: utf-8 -*-
# @author  Bin Hong

"""
"""

import os
import sys
import time
import subprocess

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
yeod.main2(poolnum = 10)

subprocess.check_output("""
    cd {local_path};
    git pull;
""".format(**{"local_path":local_path}), shell=True)

diff_num = git_diff()

if diff_num > 0 :
    subprocess.check_output("""
        cd {local_path};
        git pull;
        git commit -a -m "xxxx"; 
        git push;
    """.format(**{"local_path":local_path}), shell=True)

#! /usr/bin/env python2.7
# -*- coding: utf-8 -*-
# @author  Bin Hong

"""
"""

import sys
import os
import platform


local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..', "..")
sys.path.append(root)

from main.score import build
#from main.ta import relative

from main.work.conf import MltradeConf


def work(confer):
    assert isinstance(confer, MltradeConf)
    syms = confer.syms
    n_pool = confer.n_pool
    work2(syms, confer, n_pool)

def work2(syms, confer, n_pool):
    dir_name = confer.syms.get_name() + '_' + confer.last_trade_date
    build.work(n_pool, confer.syms.get_syms(),
                    confer, dirname = dir_name)

def work_with_original_fea(syms, confer, n_pool):
    dir_name = confer.syms.get_name() + '_' + confer.last_trade_date
    build.work(n_pool, confer.syms.get_syms(),
               confer, dirname=dir_name)
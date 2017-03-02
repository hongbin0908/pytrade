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

from main.ta import build
#from main.ta import relative

from main.work.conf import MltradeConf


def work(confer):
    assert isinstance(confer, MltradeConf)
    syms = confer.syms
    ta = confer.ta
    n_pool = confer.n_pool
    work2(syms, ta, confer, n_pool)

def work2(syms, ta, confer, n_pool):
    dir_name = confer.syms.get_name()
    if os.path.exists(confer.get_ta_file()) and not confer.force :
        print("%s exists!" % confer.get_ta_file())
        return
    #relative.work(n_pool, confer.syms.get_syms(), confer, dirname=dir_name)
    build.work(n_pool, confer.syms.get_syms(), ta,
                    confer, dirname = dir_name)
    #df.to_pickle(confer.get_ta_file())
    #df_feat.to_pickle(confer.get_feat_file())

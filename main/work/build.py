#! /usr/bin/env python2.7
# -*- coding: utf-8 -*-
# @author  Bin Hong

"""
"""

import sys
import os
import platform

from main.base.score2 import ScoreLabel

local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..', "..")
sys.path.append(root)

from main.ta import build
from main.work.conf import MltradeConf


def work(confer):
    assert isinstance(confer, MltradeConf)
    syms = confer.syms
    ta = confer.ta
    score1 = confer.score1
    score2 = confer.score2
    n_pool = confer.n_pool
    work2(syms, ta, score1, score2, confer, n_pool)

def work2(syms, ta, score1, score2, confer, n_pool):
    assert isinstance(score1, ScoreLabel)
    assert isinstance(score2, ScoreLabel)
    assert isinstance(n_pool, int)
    if os.path.exists(confer.get_ta_file()):
        print("%s exists!" % confer.get_ta_file())
        return
    print(type(confer.syms))
    dir_name = confer.syms.get_name()
    df = build.work(n_pool, confer.syms.get_syms(), ta,
                    [score1, score2], confer, dirname = dir_name)
    df.to_pickle(confer.get_ta_file())

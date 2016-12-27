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

if platform.platform().startswith("Windows"):
    TEST = True
else:
    TEST = False


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
    for symset in syms:
        out_file = os.path.join(root, "data", "ta", "%s-%s-%s.pkl"
                                % (symset.get_name(), ta.get_name(), score1.get_name()))
        if os.path.exists(out_file):
            print("%s exists!" % out_file)
            continue
        df = build.work(n_pool, symset.get_syms(), ta,
                        [score1, score2], confer)
        df.to_pickle(out_file)

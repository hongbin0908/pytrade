#! /usr/bin/env python2.7
# -*- coding: utf-8 -*-
# @author  Bin Hong

import os,sys
import pandas as pd

local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..', '..')
sys.path.append(root)

from main.model import feat_select
from maon.model import feat_select_common as fc

dataroot = os.path.join(root, "data", "feat_select")


def work(df, f):
    for i in range(10):
        frm = 50  * i
        to  = frm + 50
        setname = "sp500R%dT%d" % (frm, to)
        taname = "base1"
        (phase1, phase2, phase3) = \
            feat_select.split_dates(feat_select.load_feat(taname, setname))
        df2 = feat_select.apply(df,phase2, "label5", "_p2")
        df2 = feat_select.apply(df2,phase3, "label5", "_p3")
        feat_select.ana2(df2, f, setname)

def work2(df, f):
    setname = "sp500"
    taname = "base1"
    (phase1, phase2, phase3) = \
        feat_select.split_dates(feat_select.load_feat(taname, setname))
    df2 = feat_select.apply(df,phase2, "label5", "_p2")
    df2 = feat_select.apply(df2,phase3, "label5", "_p3")
    feat_select.ana2(df2, f, setname)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='No desc')

    parser.add_argument('--depth', dest='depth', action='store',
                        default=1, type=int)
    args = parser.parse_args()
    fout = os.path.join(dataroot,
                        "feat_select_phase1_sp500_%d_apply_future.ana" % args.depth)
    f = open(fout, "w")
    print >> f, "## sp500_base1_stable.pkl"
    df = pd.read_pickle(os.path.join(dataroot,
                              "phase1_dump",
                              "sp500_base1_%d_stable.pkl" % args.depth))

    work(df, f)
    print >> f, "## sp500_base1.pkl"
    df = pd.read_pickle(os.path.join(dataroot,
                              "phase1_dump",
                              "sp500_base1_%d.pkl" % args.depth))
    work(df, f)
    f.close()
    print fout

#! /usr/bin/env python2.7
# -*- coding: utf-8 -*-
# @author  Bin Hong
import os
import sys
import pandas as pd
from collections import Counter

local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..', '..')
sys.path.append(root)

from main.model import feat_select

dataroot = os.path.join(root, "data", "feat_select")


def cross_test(df, sets, dates, name, depth):
    f = open(os.path.join(dataroot, "feat_select_phase1_%s_%d.ana"
                          % (name, args.depth)), "w")
    feat_select.ana_fmetas(df, "base1", "sp500", f)

    abs_direct_p_set = Counter(set(df[df.direct == 1].name.unique()))
    abs_direct_n_set = Counter(set(df[df.direct == -1].name.unique()))

    orig_direct_p_set = abs_direct_p_set.copy()
    orig_direct_n_set = abs_direct_n_set.copy()
    print len(set(abs_direct_p_set))
    print len(set(abs_direct_n_set))

    abs_direct_p_set = abs_direct_p_set + abs_direct_p_set
    abs_direct_n_set = abs_direct_n_set + abs_direct_n_set
    print >>f, "="*8
    for s in sets:
        for d in dates:
            setname = s
            taname = "base1"
            filename = os.path.join(dataroot, "phase1_dump", "sp500_base1_apply_phase1_%s_%s_%d_%s_%s.pkl" %
                                   (setname, taname, args.depth, d[0], d[1])
                   )
            if not os.path.exists(filename):
                fs = feat_select.load_feat(taname, setname, d[0], d[1])
                df2 = feat_select.apply(df, fs, "label5", "_p1")
                df2.to_pickle(filename)
            df2 = pd.read_pickle(filename)
            feat_select.ana_apply(df2, "_p1", setname, f)
            cur_p_set = set(df2[df2.direct_p1 == 1].name.unique())
            cur_n_set = set(df2[df2.direct_p1 == -1].name.unique())
            abs_direct_p_set = abs_direct_p_set - Counter(set(abs_direct_p_set) - cur_p_set)
            abs_direct_n_set = abs_direct_n_set - Counter(set(abs_direct_n_set) - cur_n_set)
            print len(set(abs_direct_n_set))
    df.loc[:, "istable"] = df.apply(lambda row: 1 if row["name"] in set(abs_direct_p_set) else \
             (1 if row["name"] in set(abs_direct_n_set) else 0), axis = 1)
    df.loc[:, "direct"] = df.apply(lambda row: 0 if row["istable"] == 0 else row["direct"], axis=1)
    df.to_pickle(os.path.join(dataroot,
                                "phase1_dump",
                                "sp500_base1_%s_%d_stable.pkl" % \
                                (name,args.depth)
                             )
                 )
    print >>f, "|%d|%d|%d|" % (len(set(orig_direct_p_set)), len(set(abs_direct_p_set)), len(set(orig_direct_p_set)- set(abs_direct_p_set)))
    print >>f, "|%d|%d|%d|" % (len(set(orig_direct_n_set)), len(set(abs_direct_n_set)), len(set(orig_direct_n_set)- set(abs_direct_n_set)))
    print >> f, "## stable feats on postive direct"

    print >>f, "=" * 8, "Top feature analysis"
    for name in abs_direct_p_set:
        idx = 0
        for i, each in df[df.name == name].iterrows():
            print >>f, "|%s|%.4f|%.4f|" % (each["fname"],each["start"],each["end"])
            assert idx < 1
            idx += 1

    print >> f, "## UNstable feats on postive direct"
    for name in set(orig_direct_p_set) - set(abs_direct_p_set):
        idx = 0
        for i, each in df[df.name == name].iterrows():
            print >>f, "|%s|%.4f|%.4f|" % (each["fname"],each["start"],each["end"])
            assert idx < 1
            idx += 1

    print >> f, "## stable feats on negtive direct"
    for name in set(abs_direct_n_set):
        idx = 0
        for i, each in df[df.name == name].iterrows():
            print >>f, "|%s|%.4f|%.4f|" % (each["fname"],each["start"],each["end"])
            assert idx < 1
            idx += 1

    print >> f, "## unstable feats on negtive direct"
    for name in set(orig_direct_n_set) - set(abs_direct_n_set):
        idx = 0
        for i, each in df[df.name == name].iterrows():
            print >>f, "|%s|%.4f|%.4f|" % (each["fname"],each["start"],each["end"])
            assert idx < 1
            idx += 1

    f.close()
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='No desc')

    parser.add_argument('--depth', dest='depth', action='store',
                        default=1, type=int)
    args = parser.parse_args()
    fphase1 = os.path.join(dataroot, "phase1_dump",
                                     "sp500_base1_%d.pkl" % args.depth)
    if not os.path.exists(fphase1):
        feat_select.phase1_dump("base1", "sp500", args.depth)
    df = pd.read_pickle(fphase1)
    sets = []
    for i in range(10):
        frm = 50*i
        to = frm + 50
        sets.append("sp500R%dT%d" % (frm, to))
    print sets
    dates = []
    cross_test(df, sets, dates, "all", args.depth)
    sys.exit(0)

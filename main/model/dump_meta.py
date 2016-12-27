#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
#@author  Bin Hong

"""
"""

import sys
import os

local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..', '..')
sys.path.append(root)

from main.model import feat_select
from main.base import score
import main.base as base
import main.yeod.yeod as yeod


def extract_taname(fname):
    re = fname.split("_")[1]
    if re.startswith("ROC"):
        return "ROC"
    return re


def extract_meta(meta, thresh):
    meta = meta[meta.c_p > thresh]
    meta.sort_values("c_p", ascending=False, inplace=True)
    meta.reset_index(drop=True, inplace=True)
    lfname = list(meta["fname"])
    sfname = set([])
    lindex = []
    for i in range(len(lfname)):
        print extract_taname(lfname[i])
        if extract_taname(lfname[i]) not in sfname:
            lindex.append(i)
            sfname.add(extract_taname(lfname[i]))
            print sfname
    print lindex
    meta = meta.iloc[lindex]
    return meta


def work(setname, start, end, depth, thresh, scorename):
    """
    """
    phase1 = base.get_merged("base1",
                             getattr(yeod, "get_%s" % setname)(),
                             start, end)
    print phase1.shape
    phase1.reset_index(drop=True, inplace=True)
    phase1 = score.agn_rank_score(phase1)
    phase1 = score.agn_rank_score(phase1, interval=5, threshold=0.55)
    phase1 = score.agn_label_score(phase1, interval=5, threshold=1.0)
    meta = feat_select.flat_metas(phase1, depth, 100000, scorename)

    print meta[["fname", "c_p"]]
    meta = extract_meta(meta, thresh)
    meta.reset_index(drop=True, inplace = True)

    meta.to_pickle("./data/model/meta_base1_%s_%s_%s_%s_%d_100000.pkl" % (
        setname, scorename, start, end, depth))
    return meta


def main(args):
    return work(args.setname, args.trainstart, args.trainend,
                args.depth, args.thresh, args.scorename)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='paper test')

    parser.add_argument('--trainstart', dest="trainstart",
                        action="store", default='1984-01-01')
    parser.add_argument('--trainend', dest="trainend",
                        action="store", default='2009-12-31')
    parser.add_argument('--depth', dest="depth",
                        action="store", type=int, default=1)
    parser.add_argument('--thresh', dest="thresh",
                        action="store", type=float, default=0.52)
    parser.add_argument('--scorename', dest="scorename",
                        action="store", type=str,
                        default=score.score_name_rank(interval=5, threshold=0.55))

    parser.add_argument('setname', help = "")
    # parser.add_argument('taname', help = "")

    args = parser.parse_args()
    main(args)

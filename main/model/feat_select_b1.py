#! /usr/bin/env python2.7
# -*- coding: utf-8 -*-
# @author  Bin Hong

import os,sys
import multiprocessing

local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..', '..')
sys.path.append(root)

from main.model import feat_select



def phase1_dump(taname, setname):
    dfTa = feat_select.load_feat(taname, setname)
    (phase1, phase2, phase3) = feat_select.split_dates(dfTa)
    dfmetas = feat_select.flat_metas(feat_select.get_metas(phase1))
    outdir = os.path.join(root, "data", "feat_select", "phase1_dump")
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    dfmetas.to_pickle(os.path.join(outdir, "%s_%s.pkl" % (setname, taname)))

if __name__ == '__main__':
    pool = multiprocessing.Pool(processes = int(10) )
    results = []
    for i in range(10):
        frm = 50  * i
        to  = frm + 50
        #cmdstr = """
        #          python main/model/feat_select.py sp500R%dT%d base1
        #          """ % (frm, to)
        #print cmdstr
        setname = "sp500R%dT%d" % (frm, to)
        taname = "base1"

        results.append(pool.apply_async(phase1_dump, (taname, setname)))
    for result in results:
        print result.get()


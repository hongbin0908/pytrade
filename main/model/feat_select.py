#! /usr/bin/env python2.7
# -*- coding: utf-8 -*-
# @author  Bin Hong

"""
"""

import sys
import os
import numpy as np
import pandas as pd
import multiprocessing

local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..', '..')
sys.path.append(root)

from main.utils import time_me

import main.yeod.yeod as yeod
import main.base as base
import main.pandas_talib as pta
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import _tree

dataroot = os.path.join(root, "data", "feat_select")

def pn_ratio(df, confer):
    feat_names = base.get_feat_names(df)
    label = df.loc[:,confer.score1.get_name()]
    res = pd.DataFrame(data = None, index=feat_names)
    df = df[feat_names]
    n11 = df[label > 0.5].sum()
    n10 = df[label < 0.5].sum()

    res["n11"] = n11
    res["n10"] = n10
    res["pn_ratio"] = res["n11"]/(res["n10"]+res["n11"])
    return res

def mutual_information(df, confer):
    """
    http://nlp.stanford.edu/IR-book/html/htmledition/mutual-information-1.html#mifeatsel
    """
    feat_names = base.get_feat_names(df)
    label = df.loc[:,confer.score1.get_name()]
    res = pd.DataFrame(data = None, index=feat_names)
    df = df[feat_names]
    n11 = df[label > 0.5].sum()
    n10 = df[label < 0.5].sum()
    n01 = (1-df[label>0.5]).sum()
    n00 = (1-df[label<0.5]).sum()
    n = df.count()
    n1_ = n11+n10
    n0_ = n01+n00
    n_1 = n01+n11
    n_0 = n00+n10

    assert 0 == (n11 + n01 - df[label>0.5].count()).sum() 
    assert 0 == (n11 + n01 + n10 + n00 - df.count()).sum() 
    

    mi = n11/n*np.log2(n*n11/(n1_*n_1)) + n01/n*np.log2(n*n01/(n0_*n_1)) \
            + n10/n*np.log2(n*n10/(n1_*n_0)) + n00/n*np.log2(n*n00/(n0_*n_0))

    res["n11"] = n11
    res["n10"] = n10
    res["n01"] = n01
    res["n00"] = n00
    res["mi"] = mi
    return res

    

    

#def split_dates(df):
#    """
#    split df to [1980-2006], [2006-2011], [2011-2016]
#    """
#    phase1 = df[(df.date >= '1980-01-01') & (df.date < '2006-01-01')]
#    phase2 = df[(df.date >= '2006-01-01') & (df.date < '2011-01-01')]
#    phase3 = df[(df.date >= '2011-01-01') & (df.date < '2016-01-01')]
#    return (phase1, phase2, phase3)
#
#
#@time_me
#def load_feat(taname, setname, label, start="", end=""):
#    # get all the features
#    dfTa = base.get_merged(taname,
#                           getattr(yeod, "get_%s" % setname)(), start, end)
#    return dfTa
#
#def deep_feats(meta, threshold=0.55):
#    """
#    feat to deep analysis. c_p great than *threshold*.
#    :return: list
#    """
#    l = []
#    for i, each in meta[meta.c_p >= 0.55].iterrows():
#        l.append(each.fname)
#    return l
#
#def append_deep_feats(df, l):
#    for feat in l:
#        name = feat[3:] + "diff"
#        for i in [1, 3,5,7,14]:
#            df = df.join(pta.ROC(df, i, price = feat, name = name))
#    return df
#
#@time_me
#def flat_metas(df, depth, min_, label):
#    metas = _get_metas(df, depth, min_, label)
#    fmetas = []
#    for each in metas:
#        for i, term in enumerate(each["range"]):
#            d = {}
#            d["fname"] = each["name"]
#            d["name"] = "%s_d%d_%d" % (each["name"], depth, i)
#            d["start"] = each["range"][i][0]
#            d["end"] = each["range"][i][1]
#            d["p_chvfa"] = each["p_chvfa"][i]
#            d["n_chvfa"] = each["n_chvfa"][i]
#            d["c_p"] = each["children_p"][i]
#            d["c_n"] = each["children_n"][i]
#            d["p"] = each["p"]
#            d["n"] = each["n"]
#            #assert 1 == d["p"] + d["n"]
#            d["score"] = each["delta_impurity"]
#            d["n_samples"] = each["n_samples"][i]
#            d["direct"] = 1 if d["p_chvfa"] > 1.01  else (-1 if d["n_chvfa"] > 1.01 else 0)
#            fmetas.append(d)
#    df = pd.DataFrame(fmetas)
#    df.sort_values("c_p", ascending=False, inplace=True)
#    return df
#
#
#def flat_metas2(df, depth, min_, label):
#    metas = _get_metas2(df, depth, min_, label)
#    fmetas = []
#    for each in metas:
#        for i, term in enumerate(each["range"]):
#            d = {}
#            d["fname"] = each["name"]
#            d["name"] = "%s_d%d_%d" % (each["name"], depth, i)
#            d["start"] = each["range"][i][0]
#            d["end"] = each["range"][i][1]
#            d["p_chvfa"] = each["p_chvfa"][i]
#            d["n_chvfa"] = each["n_chvfa"][i]
#            d["c_p"] = each["children_p"][i]
#            d["c_n"] = each["children_n"][i]
#            d["p"] = each["p"]
#            d["n"] = each["n"]
#            #assert 1 == d["p"] + d["n"]
#            d["score"] = each["delta_impurity"]
#            d["n_samples"] = each["n_samples"][i]
#            d["direct"] = 1 if d["p_chvfa"] > 1.01  else (-1 if d["n_chvfa"] > 1.01 else 0)
#            fmetas.append(d)
#    df = pd.DataFrame(fmetas)
#    df.sort_values("c_p", ascending=False, inplace=True)
#    return df
#
#@time_me
#def ana_fmetas(df, taname, setname, f):
#    head = df.sort_values(["score"], ascending=False).head(40)
#    for i, each in head.iterrows():
#        print >> f, "%s,%s,%s,%s,%d,%.4f,%.4f,%d" % (each["name"], each["fname"],
#                                                     each["start"], each["end"],
#                                                     each["direct"], each["p_chvfa"], each["n_chvfa"],
#                                                     each["n_samples"])
#
#    max_score = head["score"].max()
#    mean_score = df["score"].mean()
#
#    max_p_rate = df["p_chvfa"].max()
#    mean_p_rate = df[df.direct == 1]["p_chvfa"].mean()
#
#    max_n_rate = df["n_chvfa"].max()
#    mean_n_rate = df[df.direct == -1]["n_chvfa"].mean()
#
#    direct_p_num = len(df[df.direct == 1])
#    direct_n_num = len(df[df.direct == -1])
#    direct_0_num = len(df[df.direct == 0])
#
#    print >> f, "delta_dis: |%s|%.8f|%.8f|%.4f|%.4f|%.4f|%.4f|" % (setname,
#                                                                   max_score, mean_score,
#                                                                   max_p_rate, mean_p_rate,
#                                                                   max_n_rate, mean_n_rate)
#    assert len(df) == direct_p_num + direct_n_num + direct_0_num
#    print >> f, "direct_dis: |%s|%d|%d|%d|%d|" % (setname,
#                                                  len(df), direct_p_num, direct_n_num, direct_0_num)
#
#
#def feat_broad(dfmetas, df, label):
#    print "postive ..."
#    dfp = dfmetas[(dfmetas.direct == 1) & (dfmetas.score > 0.00017)].sort_values("score", ascending=False)
#    all_ = 0
#    good = 0
#    avg = 0.0
#    for i, each in dfp.iterrows():
#        fname = each["fname"]
#        start = each["start"]
#        end = each["end"]
#        score = each["score"]
#
#        pos = df[(df[fname] >= start) & (df[fname] < end)]
#        tinpos = pos[pos[label] > 1.0]
#        rate1 = len(tinpos) * 1.0 / len(pos)
#
#        tindf = df[df[label] > 1.0]
#        rate2 = len(tindf) * 1.0 / len(df)
#        if rate1 > rate2:
#            good += 1
#        all_ += 1
#        avg += rate1
#        print 1 if rate1 > rate2 else 0, fname, start, end, score, rate1
#
#        print good, all_, avg / all_
#
#
#def exp_feat(df, name, depth=3, min_=10000):
#    tmp = df[["date", name, "score5"]]
#    tmp = tmp.replace([np.inf, -np.inf], np.nan).dropna()
#    tmp = flat_metas(tmp, depth, min_, "score5")
#    print tmp
#
#
#@time_me
#def agg_yyyy_good(df):
#    def f(x):
#        p = len([xx for xx in x.tolist() if xx > 1.0])
#        return p * 1.0 / len(x.tolist())
#
#    def f2(x):
#        return len([xx for xx in x.tolist() if xx > 1.0])
#
#    df['yyyy'] = df.date.str.slice(0, 4)
#    df['yyyyMM'] = df.date.str.slice(0, 7)
#    return df[df.pr1 == 1].groupby("yyyyMM")["label5"].agg({"p": lambda x: f(x), "count": lambda x: f2(x)}) \
#                        .sort_values("")
#
#
#@time_me
#def feat_good_label(df, meta, threshold=0.59):
#    def f5(x, lmeta1):
#        if "pr1" in x and x["pr1"] == 1:
#            return 1
#        for each in lmeta1:
#            if each[1] <= x[each[0]] < each[2]:
#                return 1
#        return 0
#
#    meta1 = meta[meta.c_p >= threshold]
#    lmeta1 = []
#    for i, each in meta1.iterrows():
#        lmeta1.append((each.fname, each.start, each.end))
#    df["pr1"] = df.apply(lambda x: f5(x, lmeta1), axis=1)
#    return df
#
#
#def apply(dfmetas, df, label, subfix):
#    fp = len(df[df[label] > 1.0]) * 1.0 / len(df)
#    fn = len(df[df[label] < 1.0]) * 1.0 / len(df)
#
#    shadows = []
#    for i, each in dfmetas.iterrows():
#        d = {}
#        d["name"] = each["name"]
#        d["fname"] = each["fname"]
#        d["start"] = each["start"]
#        d["end"] = each["end"]
#        d["p"] = fp
#        d["n"] = fn
#        dfc = df[(df[d["fname"]] >= d["start"]) & (df[d["fname"]] < d["end"])]
#        d["c_p"] = 0 if len(dfc) == 0 else len(dfc[dfc[label] > 1.0]) * 1.0 / len(dfc)
#        d["c_n"] = 0 if len(dfc) == 0 else len(dfc[dfc[label] < 1.0]) * 1.0 / len(dfc)
#        d["p_chvfa"] = d["c_p"] / d["p"]
#        d["n_chvfa"] = d["c_n"] / d["n"]
#        d["direct"] = 1 if d["p_chvfa"] > 1.01  else (-1 if d["n_chvfa"] > 1.01 else 0)
#        d["n_samples"] = len(dfc)
#        shadows.append(d)
#    df2 = pd.DataFrame(shadows)
#    return dfmetas.merge(df2, left_on=["name", "fname", "start", "end"],
#                         right_on=["name", "fname", "start", "end"],
#                         suffixes=("", subfix))
#
#
#def ana_apply(df, suffix, setname, f):
#    df1 = df[df.direct == 1]
#    rate1 = len(df1[df1["direct%s" % suffix] == 1]) * 1.0 / len(df1)
#
#    df2 = df[df.direct == -1]
#    rate2 = len(df2[df2["direct%s" % suffix] == -1]) * 1.0 / len(df2)
#
#    print >> f, "|%s|%.4f|%.4f|" % (setname, rate1, rate2)
#
#
#def ana2(df, f, setname):
#    df1 = df[df.direct == 1]
#    # for i, each in df1.iterrows():
#    #    print >> f, each["name"], each["direct"], each["direct_p2"], each["direct_p3"]
#    rate1_p2 = len(df1[df1.direct_p2 == 1]) * 1.0 / len(df1)
#    rate1_p3 = len(df1[df1.direct_p3 == 1]) * 1.0 / len(df1)
#
#    df2 = df[df.direct == -1]
#    # for i, each in df2.iterrows():
#    #    print >> f, each["name"], each["direct"], each["direct_p2"], each["direct_p3"]
#    rate2_p2 = len(df2[df2.direct_p2 == -1]) * 1.0 / len(df2)
#    rate2_p3 = len(df2[df2.direct_p3 == -1]) * 1.0 / len(df2)
#
#    print >> f, "|%s|%.4f|%.4f|%.4f|%.4f|" % (setname, rate1_p2, rate1_p3, rate2_p2, rate2_p3)
#
#
#def phase1_dump(taname, setname, depth):
#    dfTa = load_feat(taname, setname)
#    (phase1, phase2, phase3) = split_dates(dfTa)
#    dfmetas = flat_metas(phase1, depth)
#    outdir = os.path.join(root, "data", "feat_select", "phase1_dump")
#    if not os.path.exists(outdir):
#        os.makedirs(outdir)
#    dfmetas.to_pickle(os.path.join(outdir, "%s_%s_%d.pkl" % (setname, taname, depth)))
#    return dfmetas
#
#
#def _get_metas(dfTa, depth, min_, label):
#    feat_names = base.get_feat_names(dfTa)
#    idx = 0
#    results = []
#    for cur_feat in feat_names:
#        idx += 1
#        ## plen, nlen and len_ is to speed up
#        results.append(_feat_meta(cur_feat, dfTa, len(dfTa[dfTa[label] == 1]),
#                                  len(dfTa[dfTa[label] == 0]), len(dfTa), label, depth, min_))
#        print "%d done!" % idx
#    return [result for result in results]
#
#def _get_metas2(dfTa, depth, min_, label):
#    feat_names = base.get_feat_names(dfTa)
#    idx = 0
#    results = []
#    for cur_feat in feat_names:
#        idx += 1
#        ## plen, nlen and len_ is to speed up
#        results.append(_feat_meta2(cur_feat, dfTa, len(dfTa[dfTa[label] == 1]),
#                                  len(dfTa[dfTa[label] == 0]), len(dfTa), label, depth, min_))
#        print "%d done!" % idx
#    return [result for result in results]
#
#def _feat_meta2(feat, df, plen, nlen , len_, label, depth=2, min_=10000):
#    rlt = {}
#    tree = _get_tree(depth, min_)
#    npFeat = df[[feat, "ta_adx_14"]].values.copy()
#    npLabel = df[label].values.copy()
#    npLabel[npLabel > 1.0] = 1
#    npLabel[npLabel < 1.0] = 0
#
#    min_ = npFeat.min()
#    max_ = npFeat.max()
#    tree.fit(npFeat, npLabel)
#    assert isinstance(tree.tree_, _tree.Tree)
#
#    leaves = _get_leaves(tree, min_, max_)
#    rlt["splits"] = leaves
#    rlt["name"] = feat
#    rlt["p"] = 1.0 * plen / len_
#    rlt["n"] = 1.0 * nlen / len_
#    rlt["delta_impurity"] = _delta_impurity(tree, leaves)
#    rlt["impurity"] = tree.tree_.impurity[0]
#    # p1 = 1.0*len(df[df[label]>1.0])/len(df)
#    # p2 = 1.0*len(df[df[label]<1.0])/len(df)
#    # assert abs(rlt["impurity"] - (1- p1*p1 -p2*p2)) < 0.0001
#    rlt["range"] = leaves_range(leaves)
#    rlt["children_p"] = leaves_p(leaves)
#    rlt["children_n"] = [(1 - each) for each in leaves_p(leaves)]
#    rlt["p_chvfa"] = [each / rlt["p"] for each in rlt["children_p"]]
#    rlt["n_chvfa"] = [each / rlt["n"] for each in rlt["children_n"]]
#    rlt["n_samples"] = leaves_n_samples(leaves)
#
#    # for i in range(len(rlt["range"])):
#    #    cur_range = rlt["range"][i]
#    #    print feat, rlt["n_samples"][i],cur_range,len(df[(df[feat]>=cur_range[0])&(df[feat]<cur_range[1])])
#    #    assert abs(rlt["n_samples"][i] - len(df[(df[feat]>=cur_range[0])&(df[feat]<cur_range[1])])) < 1
#    return rlt
#
#def _feat_meta(feat, df, plen, nlen, len_, label, depth=1, min_=10000):
#    rlt = {}
#    tree = _get_tree(depth, min_)
#    npFeat = df[[feat]].values.copy()
#    npLabel = df[label].values.copy()
#    npLabel[npLabel > 1.0] = 1
#    npLabel[npLabel < 1.0] = 0
#
#    min_ = npFeat.min()
#    max_ = npFeat.max()
#    tree.fit(npFeat, npLabel)
#    assert isinstance(tree.tree_, _tree.Tree)
#
#    leaves = _get_leaves(tree, min_, max_)
#    rlt["splits"] = leaves
#    rlt["name"] = feat
#    rlt["p"] = 1.0 * plen / len_
#    rlt["n"] = 1.0 * nlen / len_
#    rlt["delta_impurity"] = _delta_impurity(tree, leaves)
#    rlt["impurity"] = tree.tree_.impurity[0]
#    # p1 = 1.0*len(df[df[label]>1.0])/len(df)
#    # p2 = 1.0*len(df[df[label]<1.0])/len(df)
#    # assert abs(rlt["impurity"] - (1- p1*p1 -p2*p2)) < 0.0001
#    rlt["range"] = leaves_range(leaves)
#    rlt["children_p"] = leaves_p(leaves)
#    rlt["children_n"] = [(1 - each) for each in leaves_p(leaves)]
#    rlt["p_chvfa"] = [each / rlt["p"] for each in rlt["children_p"]]
#    rlt["n_chvfa"] = [each / rlt["n"] for each in rlt["children_n"]]
#    rlt["n_samples"] = leaves_n_samples(leaves)
#
#    # for i in range(len(rlt["range"])):
#    #    cur_range = rlt["range"][i]
#    #    print feat, rlt["n_samples"][i],cur_range,len(df[(df[feat]>=cur_range[0])&(df[feat]<cur_range[1])])
#    #    assert abs(rlt["n_samples"][i] - len(df[(df[feat]>=cur_range[0])&(df[feat]<cur_range[1])])) < 1
#    return rlt
#
#
#def _get_leaves(tree, min_, max_):
#    def visit(tree, node_id, list_leaf, maximum):
#        if tree.children_left[node_id] == _tree.TREE_LEAF:
#            # current node is leaf node
#            list_leaf.append((node_id, maximum))
#        else:
#            visit(tree, tree.children_left[node_id], list_leaf,
#                  tree.threshold[node_id])
#            visit(tree, tree.children_right[node_id], list_leaf,
#                  maximum)
#
#    list_leaf = []
#    visit(tree.tree_, 0, list_leaf, np.inf)
#    assert len(list_leaf) >= 2
#    for i in range(len(list_leaf)):
#        node_id, threshold = list_leaf[i]
#        leaf = {}
#        leaf["node_id"] = node_id
#        leaf["impurity"] = tree.tree_.impurity[node_id]
#        leaf["n_samples"] = tree.tree_.n_node_samples[node_id]
#        leaf["value"] = tree.tree_.value[node_id][0]
#        p1 = leaf["value"][0] / leaf["n_samples"]
#        p2 = leaf["value"][1] / leaf["n_samples"]
#        assert abs(leaf["impurity"] - (1 - p1 * p1 - p2 * p2)) < 0.00001
#
#        if i == 0:
#            leaf["min"] = min_
#        else:
#            leaf["min"] = list_leaf[i - 1]["max"]
#        leaf["max"] = threshold
#        if np.isinf(leaf["max"]):
#            leaf["max"] = max_ + 0.0001  #
#        list_leaf[i] = leaf
#    return list_leaf
#
#
#def _delta_impurity(tree, leaves):
#    root_impurity = tree.tree_.impurity[0]
#    root_n_samples = tree.tree_.n_node_samples[0]
#    delta = 0.0
#    for leaf in leaves:
#        delta += 1.0 * leaf["n_samples"] / root_n_samples * leaf["impurity"]
#    delta = root_impurity - delta
#    return delta
#
#
#def _get_tree(depth, min_):
#    tree = DecisionTreeClassifier(min_samples_leaf=10000,
#                                  min_samples_split=min_, max_depth=depth)
#    return tree
#
#
#def leaves_n_samples(leaves):
#    n_samples = []
#    for each in leaves:
#        n_samples.append(each["n_samples"])
#    return n_samples
#
#
#def leaves_range(leaves):
#    range_ = []
#    for each in leaves:
#        range_.append((each["min"], each["max"]))
#    return range_
#
#
#def leaves_p(leaves):
#    p_ = []
#    for each in leaves:
#        p_.append(each["value"][1] / each["n_samples"])
#        assert each["value"][0] + each["value"][1] == each["n_samples"]
#    return p_

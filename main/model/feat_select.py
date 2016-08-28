#! /usr/bin/env python2.7
# -*- coding: utf-8 -*-
# @author  Bin Hong

"""
"""

import sys
import os
import numpy as np
import pandas as pd
local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..', '..')
sys.path.append(root)


import main.yeod.yeod as yeod
import main.base as base
import main.ta.build as build
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import _tree


istest = True

def split_dates(df):
    """
    split df to [1980-2000], [2000-2010], [2010-2015]
    """
    phase1 = df[(df.date >= '1980-01-01') & (df.date < '2000-01-01')]
    phase2 = df[(df.date >= '2000-01-01') & (df.date < '2010-01-01')]
    phase3 = df[(df.date >= '2010-01-01') & (df.date < '2016-01-01')]
    return (phase1, phase2, phase3)

def cal_com(df, start, end, fn, mi, ma, rate):
    num_com = 0
    num_less = 0
    lost_year = []
    for i in range(start, end):
        cur_year = get_year(df, i)
        rate_cur = acc2(cur_year, fn, mi, ma)
        if rate_cur < 0:
            num_com += 1
            num_less += 1
            continue
        if (rate_cur-1) * (rate-1) > 0:
            num_com += 1
        else:
            lost_year.append(i)
    return num_com, num_less, lost_year


def load_feat(taname, setname):
    # get all the features
    dfTa = base.get_merged(taname,
                           getattr(yeod, "get_%s" % setname)())
    dfTa = dfTa[dfTa.label5 != 1.0]
    return dfTa

def get_leaves(tree, min_, max_):
    def visit(tree, node_id, list_leaf, maximum):
        if tree.children_left[node_id] == _tree.TREE_LEAF:
            list_leaf.append((node_id, maximum))
        else:
            visit(tree, tree.children_left[node_id], list_leaf,
                  tree.threshold[node_id])
            visit(tree, tree.children_right[node_id], list_leaf,
                  maximum)
    list_leaf = []
    visit(tree.tree_, 0, list_leaf, np.inf)
    for i in range(len(list_leaf)):
        node_id, threshold = list_leaf[i]
        leaf = {}
        leaf["node_id"] = node_id
        leaf["impurity"] = tree.tree_.impurity[node_id]
        leaf["n_samples"] = tree.tree_.n_node_samples[node_id]
        leaf["value"] = tree.tree_.value[node_id][0]
        if i == 0:
            leaf["min"] = min_
        else:
            leaf["min"] = list_leaf[i-1]["max"]
        leaf["max"] = threshold
        if np.isinf(leaf["max"]):
            leaf["max"] = max_
        list_leaf[i] = leaf
    return list_leaf

def delta_impurity(tree, leaves):
    root_impurity = tree.tree_.impurity[0]
    root_n_samples = tree.tree_.n_node_samples[0]
    delta = 0.0
    for leaf in leaves:
        delta += 1.0*leaf["n_samples"]/root_n_samples*leaf["impurity"]
    delta = root_impurity - delta
    return delta

def get_tree():
    tree = DecisionTreeClassifier(min_samples_leaf=10000,
                                  min_samples_split=40000, max_depth=1)
    return tree
def leaves_n_samples(leaves):
    n_samples = []
    for each in leaves:
        n_samples.append(each["n_samples"])
    return n_samples
def leaves_range(leaves):
    range_ = []
    for each in leaves:
        range_.append((each["min"], each["max"]))
    return range_

def leaves_p(leaves):
    p_ = []
    for each in leaves:
        p_.append(each["value"][1]/each["n_samples"])
        assert each["value"][0] + each["value"][1] == each["n_samples"]
    return p_

def feat_meta(feat, df, label):
    rlt = {}
    tree = get_tree()
    npFeat = df[[feat]].values.copy()
    npLabel = df[label].values.copy()
    npLabel[npLabel > 1.0] = 1
    npLabel[npLabel < 1.0] = 0

    min_ = npFeat.min()
    max_ = npFeat.max()
    tree.fit(npFeat, npLabel)
    assert isinstance(tree.tree_, _tree.Tree)

    leaves = get_leaves(tree, min_, max_)
    rlt["splits"] = leaves
    rlt["name"] = feat
    rlt["p"] = 1.0 * len(df[df[label] > 1.0])/len(df)
    rlt["delta_impurity"] = delta_impurity(tree, leaves)
    print rlt["delta_impurity"]
    rlt["impurity"] = tree.tree_.impurity[0]
    rlt["range"] = leaves_range(leaves)
    rlt["children_p"] = leaves_p(leaves)
    rlt["p_chvfa"] = [each/rlt["p"] for each in rlt["children_p"]]
    rlt["direct"] = [1 if each > 1.01 else (-1 if each < 0.99 else 0) for each in rlt['p_chvfa']]
    rlt["n_samples"] =leaves_n_samples(leaves)
    return rlt


def get_metas(dfTa):
    feat_names = base.get_feat_names(dfTa)
    list_feat_meta = []
    idx = 0
    for cur_feat in feat_names:
        idx += 1
        if istest :
            if idx > 10:
                break
        list_feat_meta.append(feat_meta(cur_feat, dfTa, "label5"))
    return list_feat_meta

def flat_metas(metas):
    fmetas = []
    for each in metas:
        d = {}
        for i, term in enumerate(each["direct"]):
            d["fname"] = each["name"]
            d["name"] = "%s_%d" % (each["name"],i)
            d["start"] = each["range"][i][0]
            d["end"] = each["range"][i][1]
            d["direct"] = each["direct"] [i]
            d["chvfa"] = each["p_chvfa"][i]
            d["p"] = each["children_p"][i]
            d["score"] = each["delta_impurity"]
            d["n_samples"] = each["n_samples"][i]
            fmetas.append(d)
    df = pd.DataFrame(fmetas)
    df = df[["fname", "score", "direct", "chvfa", "n_samples"]]
    df.sort_values(["fname", "score"], ascending=False, inplace=True)
    print df.head(100)
    return df

def apply(dfmetas, df, label):
    fp = len(df[df[label] > 1.0]) * 1.0 / len(df)
    shadows = []
    for i, each in dfmetas.iterrows():
        d = {}
        d["featname"] =  each["fname"]
        d["start"] = each["start"]
        d["end"] = each["end"]
        dfc = df[(df[d["featname"]]>=d["start"]) & (df[d["featname"]]<d["end"])]
        d["cp"] = len(dfc[dfc[label]>1.0]) * 1.0 / len(dfc)
        d["chvfa"] = d["cp"]/fp
        shadows.append(d)
    return pd.DataFrame(shadows)

def main(args):
    dfTa = load_feat(args.taname, args.setname)
    (phase1,phase2,phase3) = split_dates(dfTa)

    dfmetas = flat_metas(get_metas(phase1))


    print apply(dfmetas, phase2, "label5")


    sys.exit(0)
    for feat in list_feat_impurity:
        for token in feat["list_leaf"]:
            fn = feat["feat_name"]
            mi = token["min"]
            ma = token["max"]
            print fn,
            print mi, ma, "\t\t",

            # print "%.4f" % (token["value"][1]*1.0/token["n_samples"]),

            rateTrain = acc2(dfTrain, fn, mi, ma)

            if rateTrain < 0:
                continue

            num_com, num_less, losts = cal_com(dfTa, 1980, 2000,
                                        fn, mi, ma, rateTrain)

            print "%d\t%d\t" % (num_com, num_less),
            num_com, num_less, losts = cal_com(dfTa, 2000, 2010,
                                        fn, mi, ma, rateTrain)
            print "%d\t%d\t" % (num_com, num_less),

            num_com, num_less, losts = cal_com(dfTa, 2010, 2015,
                                        fn, mi, ma, rateTrain)
            print "%d\t%d\t" % (num_com, num_less),
            for each in losts:
                print "%d\t" % each,
            print


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='No desc')

    #parser.add_argument('--start', dest='start',
    #                    action='store', default='1700-01-01')

    #parser.add_argument('--end',   dest='end',   action='store',
    #                    default='1999-12-31', help="model end time")

    #parser.add_argument('--label', dest='labelname', action='store',
    #                    default='label3', help="the label name")
    #parser.add_argument('--part', dest='part', action='store',
    #                    default=2, type=int)
    parser.add_argument('setname', help="setname")
    parser.add_argument('taname', help="taname")

    args = parser.parse_args()
    main(args)

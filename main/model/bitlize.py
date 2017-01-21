import os,sys
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import _tree

local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..', '..')
sys.path.append(root)

import main.base as base
from main.base.timer import Timer
from main.work import build


def feat_select(df, split_point, label, depth, min_, n_pool):
    df_len = len(df)
    split_point = int(df_len * split_point)
    df1 = df.iloc[:split_point]
    df2 = df.iloc[split_point:]
    assert df_len == len(df1) + len(df2)
    with Timer("bit1 bitlize") as t:
        df_bit1 = bitlize(df1, label, depth, min_, n_pool)
    with Timer("bit2 apply") as t:
        df_bit2 = apply(df_bit1, df2, label)
    df_merged = df_bit1.merge(df_bit2, on = ["name"], suffixes=("", "_df2"))
    assert len(df_bit1) == len(df_bit2) == len(df_merged)
    assert 0 == (df_merged.start - df_merged.start_df2).sum()
    assert 0 == (df_merged.end - df_merged.end_df2).sum()
    df_merged["direct"] = df_merged.apply(lambda row: 1 if row["p_chvfa"] > 1.0 and row["p_chvfa_df2"] > 1.0 else 0, axis = 1)
    df_all = apply(df_bit1, df, label)
    assert len(df_all) == len(df_merged)
    df_all["direct"] = df_merged["direct"]
    return df_all[df_all.direct == 1]


def apply(dfBit, df, label):
    df_len = len(df)
    fp = len(df[df[label] == 1.0]) * 1.0 / df_len
    fn = len(df[df[label] == 0.0]) * 1.0 / df_len
    assert 1 == fp + fn

    shadows = []
    for i, each in dfBit.iterrows():
        d = {}
        d["name"] = each["name"]
        d["fname"] = each["fname"]
        d["start"] = each["start"]
        d["end"] = each["end"]
        d["p"] = fp
        d["n"] = fn
        dfc = df[(df[d["fname"]] >= d["start"]) & (df[d["fname"]] < d["end"])]
        d["c_p"] = 0 if len(dfc) == 0 else len(dfc[dfc[label] == 1.0]) * 1.0 / len(dfc)
        d["c_n"] = 0 if len(dfc) == 0 else len(dfc[dfc[label] == 0.0]) * 1.0 / len(dfc)
        d["p_chvfa"] = d["c_p"] / d["p"]
        d["n_chvfa"] = d["c_n"] / d["n"]
        d["n_samples"] = len(dfc)
        shadows.append(d)
    df2 = pd.DataFrame(shadows)
    return df2

def bitlize(df, label, depth, min_, n_pool):
    """
    transform the feature into binary form.
    :param df: the original features
    :param depth: the depth of training tree.
                  the result features length is len(df) * 2^(depth)
    :param min_:  the min sample num of each feature.
    :param label: the crition label
    :return: the new feature dataframe
    """
    metas = _get_metas(df, depth, min_, label, n_pool)
    fmetas = []
    for each in metas:
        for i, term in enumerate(each["range"]):
            d = {}
            d["fname"] = each["name"]
            d["name"] = "%s_d%d_%d" % (each["name"], depth, i)
            d["start"] = each["range"][i][0]
            d["end"] = each["range"][i][1]
            d["p_chvfa"] = each["p_chvfa"][i]
            d["n_chvfa"] = each["n_chvfa"][i]
            d["c_p"] = each["children_p"][i]
            d["c_n"] = each["children_n"][i]
            d["p"] = each["p"]
            d["n"] = each["n"]
            assert 1 == d["p"] + d["n"]
            d["score"] = each["delta_impurity"]
            d["n_samples"] = each["n_samples"][i]
            # d["direct"] = 1 if d["p_chvfa"] > 1.0 else 0
            fmetas.append(d)
    df = pd.DataFrame(fmetas)
    df.sort_values("c_p", ascending=False, inplace=True)
    return df


def _get_metas(dfTa, depth, min_, label, n_pool):
    feat_names = base.get_feat_names(dfTa)
    idx = 0
    results = []
    import concurrent.futures
    if n_pool == 1:
        for cur_feat in feat_names:
            feat_meta = _feat_meta(cur_feat, dfTa, len(dfTa[dfTa[label] == 1]),
                                      len(dfTa[dfTa[label] == 0]), len(dfTa), label, depth, min_)
            if None != feat_meta:
                results.append(feat_meta)
    else:
        Executor = concurrent.futures.ProcessPoolExecutor
        plen = len(dfTa[dfTa[label] == 1])
        nlen = len(dfTa[dfTa[label] == 0])
        alen = len(dfTa)
        with Executor(max_workers=n_pool) as executor:
            futures = {executor.submit(_feat_meta, cur_feat, dfTa[[cur_feat, label]].copy(), plen, nlen, alen, label, depth, min_): cur_feat for cur_feat in feat_names}
            for future in concurrent.futures.as_completed(futures):
                try:
                    cur_feat = futures[future]
                    results.append(future.result())
                except Exception as exc:
                    import traceback
                    traceback.print_exc()
                    sys.exit(1)

    return results


def _feat_meta(feat, df, plen, nlen, len_, label, depth=1, min_=10000):
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    rlt = {}
    tree = _get_tree(depth, min_)
    npFeat = df[[feat]].values.copy()
    npLabel = df[label].values.copy()
    npLabel[npLabel >= 1.0] = 1
    npLabel[npLabel < 1.0] = 0

    min_ = npFeat.min()
    max_ = npFeat.max()
    tree.fit(npFeat, npLabel)
    assert isinstance(tree.tree_, _tree.Tree)
    leaves = _get_leaves(tree, min_, max_)
    if None == leaves:
        return None
    rlt["splits"] = leaves
    rlt["name"] = feat
    rlt["p"] = 1.0 * plen / len_
    rlt["n"] = 1.0 * nlen / len_
    rlt["delta_impurity"] = _delta_impurity(tree, leaves)
    rlt["impurity"] = tree.tree_.impurity[0]
    # p1 = 1.0*len(df[df[label]>1.0])/len(df)
    # p2 = 1.0*len(df[df[label]<1.0])/len(df)
    # assert abs(rlt["impurity"] - (1- p1*p1 -p2*p2)) < 0.0001
    rlt["range"] = leaves_range(leaves)
    rlt["children_p"] = leaves_p(leaves)
    rlt["children_n"] = [(1 - each) for each in leaves_p(leaves)]
    rlt["p_chvfa"] = [each / rlt["p"] for each in rlt["children_p"]]
    rlt["n_chvfa"] = [each / rlt["n"] for each in rlt["children_n"]]
    rlt["n_samples"] = _leaves_n_samples(leaves)

    # for i in range(len(rlt["range"])):
    #    cur_range = rlt["range"][i]
    #    print feat, rlt["n_samples"][i],cur_range,len(df[(df[feat]>=cur_range[0])&(df[feat]<cur_range[1])])
    #    assert abs(rlt["n_samples"][i] - len(df[(df[feat]>=cur_range[0])&(df[feat]<cur_range[1])])) < 1
    return rlt


def _leaves_n_samples(leaves):
    n_samples = []
    for each in leaves:
        n_samples.append(each["n_samples"])
    return n_samples


def _get_leaves(tree, min_, max_):
    def visit(tree, node_id, list_leaf, maximum):
        if tree.children_left[node_id] == _tree.TREE_LEAF:
            # current node is leaf node
            list_leaf.append((node_id, maximum))
        else:
            visit(tree, tree.children_left[node_id], list_leaf,
                  tree.threshold[node_id])
            visit(tree, tree.children_right[node_id], list_leaf,
                  maximum)

    list_leaf = []
    visit(tree.tree_, 0, list_leaf, np.inf)
    assert len(list_leaf) >0
    for i in range(len(list_leaf)):
        node_id, threshold = list_leaf[i]
        leaf = {}
        leaf["node_id"] = node_id
        leaf["impurity"] = tree.tree_.impurity[node_id]
        leaf["n_samples"] = tree.tree_.n_node_samples[node_id]
        leaf["value"] = tree.tree_.value[node_id][0]
        p1 = leaf["value"][0] / leaf["n_samples"]
        p2 = leaf["value"][1] / leaf["n_samples"]
        assert abs(leaf["impurity"] - (1 - p1 * p1 - p2 * p2)) < 0.00001

        if i == 0:
            leaf["min"] = min_
        else:
            leaf["min"] = list_leaf[i - 1]["max"]
        leaf["max"] = threshold
        if np.isinf(leaf["max"]):
            leaf["max"] = max_ + 0.0001  #
        list_leaf[i] = leaf
    return list_leaf


def _delta_impurity(tree, leaves):
    root_impurity = tree.tree_.impurity[0]
    root_n_samples = tree.tree_.n_node_samples[0]
    delta = 0.0
    for leaf in leaves:
        delta += 1.0 * leaf["n_samples"] / root_n_samples * leaf["impurity"]
    delta = root_impurity - delta
    return delta


def leaves_range(leaves):
    range_ = []
    for each in leaves:
        range_.append((each["min"], each["max"]))
    return range_

def leaves_p(leaves):
    p_ = []
    for each in leaves:
        p_.append(each["value"][1] / each["n_samples"])
        assert each["value"][0] + each["value"][1] == each["n_samples"]
    return p_

def _get_tree(depth, min_):
    tree = DecisionTreeClassifier(min_samples_leaf=min_,
                                  min_samples_split=min_, max_depth=depth)
    return tree


if __name__ == "__main__":
    from run import run
    confer = run.getConf()
    build.work(confer)
    df = pd.read_pickle(os.path.join(root, "data", "ta",
                                     "sp500w5i0-TaBase1Ext4El-score_label_5_100.pkl"))
    feat_select(df, 0.8, confer.score1.get_name(), 1, 100)
    #dfMetas = bitlize(df, confer.score1.get_name(),1, 100)
    #print(dfMetas.sort_values("p_chvfa", ascending=False)[["name","fname", "p_chvfa","n_chvfa", "p"]].head(10))
    #print(dfMetas.sort_values("n_chvfa", ascending=False)[["name","fname", "n_chvfa","p_chvfa", "p"]].head(10))



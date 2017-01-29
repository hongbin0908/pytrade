#! /usr/bin/env python2.7
# -*- coding: utf-8 -*-
# @author  Bin Hong

"""
"""

import sys
import os
import numpy as np
import pandas as pd
import ntpath

local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..', "..")
sys.path.append(root)

import main.base as base
from main.model import model_work
from main.model import ana
from main.work.conf import MltradeConf
from sklearn.base import clone
from main.base import decision_path
from main.model.post import CrosserSet

def work(confer):
    def accurate(crosses, crossname):
        print('\n\n|symset|glo_l1|sel_l1|glo_l2|sel_l2|select_len|min|max|', file=out_file)
        print('|------|------|------|------|------|----------|---|---|', file=out_file)
        for threshold in [2000, 1000, 500, 200, 100]:
            for each in crosses:
                cross = each[crossname]
                df_test = pd.concat([c.df_test for c in cross])
                df_test.sort_values("pred", ascending=False, inplace=True)
                df_select, df_year, df_month = ana.select2( confer.score1, confer.score2,
                                                            df_test,
                                                            2, threshold)
                print('|%s|%.2f|%.2f|%.2f|%.2f|%d|%.2f|%.2f|'
                      % (
                          each["symsetname"],
                          ana.accurate(df_test, confer.score1),
                          ana.accurate(df_select, confer.score1),
                          ana.accurate(df_test, confer.score2),
                          ana.accurate(df_select , confer.score2),
                          len(df_select),
                          df_select.tail(1)["pred"] if len(df_select) > 0 else 0,
                          df_select.head(1)["pred"] if len(df_select) > 0 else 0), file=out_file)

    def top_10_decision_path(crosses, crossname):
        print('|symset|glo_l1|sel_l1|glo_l2|sel_l2|select_len|min|max|', file=out_file)
        print('|------|------|------|------|------|----------|---|---|', file=out_file)
        for each in crosses:
            cross = each[crossname]
            df_test = pd.concat([c.df_test for c in cross])
            df_select, df_year, df_month = ana.select2(confer.score1, confer.score2, df_test,
                                                       1, 10)
            print('|%s|%.2f|%.2f|%.2f|%.2f|%d|%.2f|%.2f|'
                  % (
                      each["symsetname"],
                      ana.accurate(df_test, confer.score1),
                      ana.accurate(df_select, confer.score1),
                      ana.accurate(df_test, confer.score2),
                      ana.accurate(df_select , confer.score2),
                      len(df_select),
                      df_select.tail(1)["pred"] if len(df_select) > 0 else 0,
                      df_select.head(1)["pred"] if len(df_select) > 0 else 0), file=out_file)
            np_feat = df_select[base.get_feat_names(df_select)].values
            classifier = cross[0].classifier
            for i in range(len(np_feat)):
                x = np_feat[i,:]
                print(x)
                dot_file = os.path.join(root, "data", "cross",
                                        'top_10_decision_path-%s-%d'
                                        % (each["symsetname"], i))
                decision_path.export_decision_path2(classifier, x, dot_file + ".dot" ,
                                                    feature_names=base.get_feat_names(df_select))
                import pydot
                (graph,) = pydot.graph_from_dot_file(dot_file + ".dot")
                graph.write_png(dot_file + ".png")
        for each in crosses:
            for i in range(10):
                dot_file = os.path.join(root, "data", "cross",
                                        'top_10_decision_path-%s-%d'
                                        % (each["symsetname"], i))
                print("![](%s.png)" % (dot_file), file=out_file)

    assert isinstance(confer, MltradeConf)
    out_file_name = confer.get_out_file_prefix() + ".model.md"
    print(out_file_name)
    out_file = open(out_file_name, "w", encoding="utf-8")
    stuff_dir_name = out_file_name + ".data"
    os.makedirs(stuff_dir_name, exist_ok=True)

    crosser_set = CrosserSet(confer)
    print("\n" + crosser_set.to_table("model").round(4).to_html(), file=out_file)
    crosser_set.plot_roc("model",
                     os.path.join(stuff_dir_name, "model.png"))
    print("\n![](./%s.data/%s.png)" % (ntpath.basename(out_file_name), "model"), file=out_file)

    print("\n"+ crosser_set.to_table("valid").round(4).to_html(), file=out_file)
    crosser_set.plot_roc("valid",
                     os.path.join(stuff_dir_name, "valid.png"))
    print("\n![](./%s.data/%s.png)" % (ntpath.basename(out_file_name), "valid"), file=out_file)

    print("## bulls...", file=out_file)
    crosser_set.plot_precision_recall_bulls("model",
                         os.path.join(stuff_dir_name, "model_pp.png"))
    print("\n![](./%s.data/%s.png)" % (ntpath.basename(out_file_name), "model_pp"), file=out_file)
    print("\n"+ crosser_set.to_table("valid").round(4).to_html(), file=out_file)

    crosser_set.plot_precision_recall_bulls("valid",
                         os.path.join(stuff_dir_name, "valid_pp.png"))
    print("\n![](./%s.data/%s.png)" % (ntpath.basename(out_file_name), "valid_pp"), file=out_file)
    print("\n" + crosser_set.ipts_table("model").head(10).round(4).to_html(), file=out_file)
    print("\n" + crosser_set.ipts_table("model").tail(10).round(4).to_html(), file=out_file)

    print("## bears...", file=out_file)
    crosser_set.plot_precision_recall_bears("model",
                                      os.path.join(stuff_dir_name, "model_pp_bears.png"))
    print("\n![](./%s.data/%s.png)" % (ntpath.basename(out_file_name), "model_pp_bears"), file=out_file)
    print("\n"+ crosser_set.to_table("valid").round(4).to_html(), file=out_file)

    crosser_set.plot_precision_recall_bears("valid",
                                      os.path.join(stuff_dir_name, "valid_pp_bears.png"))
    print("\n![](./%s.data/%s.png)" % (ntpath.basename(out_file_name), "valid_pp_bears"), file=out_file)
    print("\n" + crosser_set.ipts_table("model").head(10).round(4).to_html(), file=out_file)
    print("\n" + crosser_set.ipts_table("model").tail(10).round(4).to_html(), file=out_file)
    print("\n" + crosser_set.accurate("model").round(4).to_html(), file=out_file)

    crosser_set.plot_top_precision("model",
                         os.path.join(stuff_dir_name, "model_tp.png"), score_name="pred")
    print("\n![](./%s.data/%s.png)" % (ntpath.basename(out_file_name), "model_tp"), file=out_file)
    crosser_set.plot_top_precision("valid",
                         os.path.join(stuff_dir_name, "valid_tp.png"), score_name="pred")
    print("\n![](./%s.data/%s.png)" % (ntpath.basename(out_file_name), "valid_tp"), file=out_file)

    crosser_set.plot_top_precision("model",
                                   os.path.join(stuff_dir_name, "model_tp2.png"), score_name="pred2")
    print("\n![](./%s.data/%s.png)" % (ntpath.basename(out_file_name), "model_tp2"), file=out_file)
    crosser_set.plot_top_precision("valid",
                                   os.path.join(stuff_dir_name, "valid_tp2.png"), score_name="pred2")
    print("\n![](./%s.data/%s.png)" % (ntpath.basename(out_file_name), "valid_tp2"), file=out_file)
    print("\n" + crosser_set.top_bulls("model").round(4).to_html(), file=out_file)
    print("\n" + crosser_set.top_bulls("valid").round(4).to_html(), file=out_file)

    print("\n" + crosser_set.top_bears("model").round(4).to_html(), file=out_file)
    print("\n" + crosser_set.top_bears("valid").round(4).to_html(), file=out_file)
    out_file.close()

    import markdown2 as md
    text = ""
    with open(out_file_name, 'r', encoding='utf-8') as f:
        text = f.read()
    html = md.markdown(text, extras=["tables"])
    out_file_html = confer.get_out_file_prefix() + ".model.html"
    with open(out_file_html, "w", encoding='utf-8') as fout:
        print(html, file=fout)

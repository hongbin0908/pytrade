#! /usr/bin/env python2.7
# -*- coding: utf-8 -*-
# @author  Bin Hong

"""
"""

import sys
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
import platform
import ntpath

local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..', "..")
sys.path.append(root)

import main.base as base
from main.model import pred
from main.work.conf import MltradeConf
from main.model.post import CrosserSet

def work(confer, last_date):
    out_file_name = confer.get_out_file_prefix() + ".pred.md"
    out_file = open(out_file_name, "w", encoding="utf-8")
    stuff_dir_name = out_file_name + ".data"
    os.makedirs(stuff_dir_name, exist_ok=True)

    crosser_set = CrosserSet(confer)

    print("\n" + crosser_set.pred(last_date).round(4).to_html(), file=out_file)

    import markdown2 as md
    text = ""
    with open(out_file_name, 'r', encoding='utf-8') as f:
        text = f.read()
    html = md.markdown(text, extras=["tables"])
    out_file_html = confer.get_out_file_prefix() + ".pred.html"
    with open(out_file_html, "w", encoding='utf-8') as fout:
        print(html, file=fout)

    from shutil import copyfile
    copyfile(out_file_html, os.path.join(root, "report", "pred.html"))

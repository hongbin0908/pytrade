#!/usr/bin/env python
# -*- coding: utf-8 -*-
#@author  Bin Hong

import sys,os
import json
import numpy as np
import pandas as pd
import math
import multiprocessing
import cPickle as pkl
from sklearn.externals import joblib # to dump model
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
import pydot
local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..', '..')
sys.path.append(root)
sys.path.append(local_path)

import main.model.modeling as  model
import main.pred.pred as pred
import main.base as base

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5 (except OSError, exc: for Python <2.5)
            pass
def main(argv):
    clsName = argv[1]
    cls = joblib.load(clsName)
    idx = 0
    out_dir = os.path.join(root, 'data', 'graph',  "tmp")
    mkdir_p(out_dir)
    print out_dir
    dot_data = StringIO()
    ta = pd.read_pickle(argv[2])
    names = base.get_feat_names(ta)
    for estimator in cls.estimators_:
        dotfile = os.path.join(out_dir, '%d.dot' % idx)
        export_graphviz(estimator, feature_names = names, out_file=os.path.join(out_dir, '%d.dot' % idx))
        #graph = pydot.graph_from_dot_file(dotfile)
        #graph.write_pdf(os.path.join(out_dir, "%d.pdf" % idx))
        #Image(graph.create_png())
        idx += 1
if __name__ == '__main__':
    main(sys.argv)

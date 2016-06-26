#!/usr/bin/env python
# -*- coding: utf-8 -*-

#@author  Bin Hong

import sys,os
local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..')
sys.path.append(root)

import model.model_param_set as param_set

"""
## 研究目的
## 结论
"""
ta =  param_set.d_dir_ta

#params = ("tadowcall1_GBCv1n1000md3lr001_l5_s1700e2009" , ta["tadowcall1"], "label5", ("2010-01-01", '2016-12-31'),  200)
params = ("call1s1_dow_GBCv1n322md3lr001_l5_s1700e2009" , ta["call1s1_dow"], "label5", ("2010-01-01", '2016-12-31'),  200)


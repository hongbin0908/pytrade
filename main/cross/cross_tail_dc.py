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
    研究模型在时间上的稳定性
## 结论
"""
ta =  param_set.d_dir_ta

l_params = [
        ("taselect_GBCv1n1000md3lr001_l5_s2006e2015" , ta["taselect"], "label5", ("2016-01-01", '2016-12-31'),  100),
        ("taselect_GBCv1n1000md3lr001_l5_s2006e2015" , ta["taselect"], "label5", ("2016-01-01", '2016-12-31'),  200),
        ("taselect_GBCv1n1000md3lr001_l5_s2006e2015" , ta["taselect"], "label5", ("2016-01-01", '2016-12-31'),  300),
        ("taselect_GBCv1n1000md3lr001_l5_s2006e2015" , ta["taselect"], "label5", ("2016-01-01", '2016-12-31'),  400),
        ("taselect_GBCv1n1000md3lr001_l5_s2006e2015" , ta["taselect"], "label5", ("2016-01-01", '2016-12-31'),  500),
        ("taselect_GBCv1n1000md3lr001_l5_s2006e2015" , ta["taselect"], "label5", ("2016-01-01", '2016-12-31'),  -1),

        ("ta1_GBCv1n1000md3lr001_l5_s2006e2015" , ta["ta1"], "label5", ("2016-01-01", '2016-12-31'),  100),
        ("ta1_GBCv1n1000md3lr001_l5_s2006e2015" , ta["ta1"], "label5", ("2016-01-01", '2016-12-31'),  200),
        ("ta1_GBCv1n1000md3lr001_l5_s2006e2015" , ta["ta1"], "label5", ("2016-01-01", '2016-12-31'),  300),
        ("ta1_GBCv1n1000md3lr001_l5_s2006e2015" , ta["ta1"], "label5", ("2016-01-01", '2016-12-31'),  400),
        ("ta1_GBCv1n1000md3lr001_l5_s2006e2015" , ta["ta1"], "label5", ("2016-01-01", '2016-12-31'),  500),
        ("ta1_GBCv1n1000md3lr001_l5_s2006e2015" , ta["ta1"], "label5", ("2016-01-01", '2016-12-31'),  -1),
        ]


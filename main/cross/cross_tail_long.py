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

l_params = [
        ("ta1_GBCv1n1000md3lr001_l5_s1700e2009" , ta["ta1"], "label5", ("2010-01-01", '2010-12-31'),  500),
        ("ta1_GBCv1n1000md3lr001_l5_s1700e2009" , ta["ta1"], "label5", ("2010-01-01", '2010-12-31'),  -1),

        ("ta1_GBCv1n1000md3lr001_l5_s1700e2009" , ta["ta1"], "label5", ("2011-01-01", '2011-12-31'),  500),
        ("ta1_GBCv1n1000md3lr001_l5_s1700e2009" , ta["ta1"], "label5", ("2011-01-01", '2011-12-31'), -1),


        ("ta1_GBCv1n1000md3lr001_l5_s1700e2009" , ta["ta1"], "label5", ("2012-01-01", '2012-12-31'),  500),
        ("ta1_GBCv1n1000md3lr001_l5_s1700e2009" , ta["ta1"], "label5", ("2012-01-01", '2012-12-31'),  -1),


        ("ta1_GBCv1n1000md3lr001_l5_s1700e2009" , ta["ta1"], "label5", ("2013-01-01", '2013-12-31'),  500),
        ("ta1_GBCv1n1000md3lr001_l5_s1700e2009" , ta["ta1"], "label5", ("2013-01-01", '2013-12-31'),  -1),


        ("ta1_GBCv1n1000md3lr001_l5_s1700e2009" , ta["ta1"], "label5", ("2014-01-01", '2014-12-31'),  500),
        ("ta1_GBCv1n1000md3lr001_l5_s1700e2009" , ta["ta1"], "label5", ("2014-01-01", '2014-12-31'),  -1),

        ("ta1_GBCv1n1000md3lr001_l5_s1700e2009" , ta["ta1"], "label5", ("2015-01-01", '2015-12-31'),  500),
        ("ta1_GBCv1n1000md3lr001_l5_s1700e2009" , ta["ta1"], "label5", ("2015-01-01", '2015-12-31'),  -1),


        ("ta1_GBCv1n1000md3lr001_l5_s1700e2009" , ta["ta1"], "label5", ("2016-01-01", '2016-12-31'),  500),
        ("ta1_GBCv1n1000md3lr001_l5_s1700e2009" , ta["ta1"], "label5", ("2016-01-01", '2016-12-31'),  -1),

        ]


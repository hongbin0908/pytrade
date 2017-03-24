#! /usr/bin/env python2.7
# -*- coding: utf-8 -*-
# @author  Bin Hong
import os,sys
import zipfile
local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..')
sys.path.append(root)

from main import base

base.zip_folder(os.path.join(root, "demo", "zipfile_dir"), 
        os.path.join(root, "demo", "zipfile_dir.zip"))

zipfile.ZipFile(os.path.join(root, "demo", "zipfile_dir.zip")).extractall(os.path.join(root, 'demo', "extract_dir"))
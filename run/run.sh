#!/bin/sh
############################
#@author hongbin@youzan.com
#@date
#@desc TODO
############################
export PATH=/usr/bin:$PATH
export SCRIPT_PATH=`dirname $(readlink -f $0)` # get the path of the script
pushd . > /dev/null
cd "$SCRIPT_PATH"/../

rm -rf data/ta/*
rm -rf data/cross/*
python3 run/get_yeod.py
python3 run/run.py

git add report/*
git commit -a -m "running ..."
git push

 
popd  > /dev/null # return the directory orignal

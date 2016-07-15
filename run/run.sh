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

yestday=$1
 
#rm -rf ./data/yeod/*
./main/yeod/yeod.py  sp500Top50  1
./main/ta/build.py  sp500Top50  call1s1 1

popd  > /dev/null # return the directory orignal

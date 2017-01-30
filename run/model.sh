#!/usr/bin/env bash


setname=sp500
taname=stable_phase1_3
clsname=RFCv1n2000md4msl1000
python3 ./main/yeod/yeod.py ${setname}  -p10
python3 ./main/yeod/yeod.py index_dow  -p1


python3 ./main/ta/build.py ${setname} base1  -p10
python ./main/model/dump_meta.py ${setname} --depth=1 --thresh=0.46
python ./main/model/dump_meta.py ${setname} --depth=2 --thresh=0.48
python ./main/model/dump_meta.py ${setname} --depth=3 --thresh=0.50

./main/ta/build.py ${setname} ${taname}  -p1
./main/model/model_work.py  ${setname} ${taname} ${clsname}

./main/paper/paper.py  `./tool/model_name.py ${setname} ${taname} ${clsname}`  ${setname} ${taname} --start="1984-01-01" --end="2009-12-31"
./main/paper/paper.py  `./tool/model_name.py ${setname} ${taname} ${clsname}`  ${setname} ${taname} --start="2010-01-01" --end="2016-12-31"

./main/paper/ana.py  ./data/paper/sp500Top50-stable_phase1_3-${clsname}-score5-1984-01-01-2009-12-31.pklsp500Top50-2010-01-01-2016-12-31.pre.csv  --thresh=0.56
./main/paper/ana.py  ./data/paper/sp500Top50-stable_phase1_3-${clsname}-score5-1984-01-01-2009-12-31.pklsp500Top50-1984-01-01-2009-12-31.pre.csv  --thresh=0.57


./main/pred/pred.py `./tool/model_name.py ${setname} ${taname} ${clsname}` ${setname} ${taname} "2016-09-23"
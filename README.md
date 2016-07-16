## pytrade

my trading system depending on machine learning(gbdc)


## install

1. install CPython

    sudo easy_install-2.7 cython

2. install TA-Lib

    wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz

    untar and cd

    ./configure --prefix=/usr

    make

    sudo make install

3.  sudo pip2.7 install TA-Lib
    
4.  sudo pip2.7 install numpy
    
5.  sudo pip2.7 install pandas
    
6.  sudo pip2.7 install scikit-learn

## usage

### get data
1. retrival end of day stock data from Yahoo! finance API. 
    ./main/yeod/yeod_b.py sp500Top100  50 10
2. calate Tachnichal Analysis Data of each splits of Stock Data
    ./main/ta/build_b.py sp500Top100 50 call1s4 10

### modeling
1. create model using 1700 ~ 2009 stock data
    ./main/model/model_work_b.py  call1s4-sp500Top100  50  GBCv1n1000md3lr001 label5 1700-01-01 2009-12-31 0 0 10
### paper test
1. test the model using 2010 ~ 2016
    ./main/paper/paper_b.py  model, 650, "%s-%s" % (ta,eod), batch, "2010-06-01", "2016-06-31", 2, 400])

### preding
    pred_b.main([model, 650, "%s-%s-%d" % (ta, eod, batch),
        last_date, last_date, "label5"])




    ./run/run.py


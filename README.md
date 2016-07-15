## pytrade

my trading system depending on machine learning


## install

1. install CPython

    sudo easy_install-2.7 cython

2. install TA-Lib

    wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz

    untar and cd

    ./configure --prefix=/usr

    make

    sudo make install

    sudo pip2.7 install TA-Lib
    
    sudo pip2.7 install numpy
    
    sudo pip2.7 install pandas
    
    sudo pip2.7 install scikit-learn

## usage

### train

./main/yeod/yeod.py  sp500Top50 1
./main/ta/build.py  sp500Top50 call1s1 1 

### modeling
./main/model/model_work.py  call1s1_sp500Top50 GBCv1n500md3lr001 label5 1700-01-01 2009-12-31 0

### select stage



    ./run/run.py


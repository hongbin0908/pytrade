import talib
import numpy as np
    

def PRICE_n(close, n):
    np_pre_closes = np.array(close)
    for i in range(n):
        pre_closes.append(np.array)
    for i in range(n, close.shape[0]):
        pre_closes.append(close[i])
    np_pre_closes = np.array(pre_closes)
    assert(np_pre_closes.shape == close.shape)
    return np_pre_closes
def PRICE_0(close):
    return PRICE_n(close, 0)
def PRICE_1(close):
    return PRICE_n(close, 1)
def PRICE_2(close):
    return PRICE_n(close, 2)
def PRICE_3(close):
    return PRICE_n(close, 3)
def PRICE_4(close):
    return PRICE_n(close, 4)
def PRICE_5(close):
    return PRICE_n(close, 5)
def PRICE_6(close):
    return PRICE_n(close, 6)
def PRICE_7(close):
    return PRICE_n(close, 7)
def PRICE_8(close):
    return PRICE_n(close, 8)
def PRICE_9(close):
    return PRICE_n(close, 9)
def ADX_extn(high, low, close, n):
    adx = talib.ADX(high, low,close)
    assert  adx.shape[0] > 2 * n
    for i in range(adx.shape[0]-n):
        idx = adx.shape[0]-1 -i
        pre = adx[idx-n]
        if pre != np.NaN:
            adx[idx] = adx[idx] - adx[idx-n]
        else:
            adx[idx] = np.NaN
    return adx;
def ADX_ext1(high, low, close):
    return ADX_extn(high, low, close, 1)
def ADX_ext2(high, low, close):
    return ADX_extn(high, low, close, 2)
def ADX_ext3(high, low, close):
    return ADX_extn(high, low, close, 3)
def ADX_ext4(high, low, close):
    return ADX_extn(high, low, close, 4)
def ADX_ext5(high, low, close):
    return ADX_extn(high, low, close, 5)
def ADX_ext10(high, low, close):
    return ADX_extn(high, low, close, 10)
def ADX_ext30(high, low, close):
    return ADX_extn(high, low, close, 30)

def ADX_ext20n(high, low, close, n):
    mdi = talib.MINUS_DI(high, low, close)
    pdi = talib.PLUS_DI(high, low, close)
    adx = talib.ADX(high, low, close)
    tmp1 = pdi - mdi
    for i in range(tmp1.shape[0] - n):
        idx = tmp1.shape[0]-1-i
        tmp1[idx] = tmp1[idx] - tmp1[idx-n]
    return tmp1
def ADX_ext201(high, low, close):
    return ADX_ext20n(high, low, close, 1)
def ADX_ext202(high, low, close):
    return ADX_ext20n(high, low, close, 2)
def ADX_ext203(high, low, close):
    return ADX_ext20n(high, low, close, 3)
def ADX_ext204(high, low, close):
    return ADX_ext20n(high, low, close, 4)
def ADX_ext205(high, low, close):
    return ADX_ext20n(high, low, close, 5)
def ADX_ext210(high, low, close):
    return ADX_ext20n(high, low, close, 10)
def ADX_ext230(high, low, close):
    return ADX_ext20n(high, low, close, 30)


def ADX_ext30n(high, low, close, n):
    mdi = talib.MINUS_DI(high, low, close)
    pdi = talib.PLUS_DI(high, low, close)
    adx = talib.ADX(high, low, close)
    tmp1 = adx -pdi
    for i in range(tmp1.shape[0] - n):
        idx = tmp1.shape[0]-1-i
        tmp1[idx] = tmp1[idx] - tmp1[idx-n]
    return tmp1
def ADX_ext301(high, low, close):
    return ADX_ext30n(high, low, close, 1)
def ADX_ext302(high, low, close):
    return ADX_ext30n(high, low, close, 2)
def ADX_ext303(high, low, close):
    return ADX_ext30n(high, low, close, 3)
def ADX_ext304(high, low, close):
    return ADX_ext30n(high, low, close, 4)
def ADX_ext305(high, low, close):
    return ADX_ext30n(high, low, close, 5)
def ADX_ext310(high, low, close):
    return ADX_ext30n(high, low, close, 10)
def ADX_ext330(high, low, close):
    return ADX_ext30n(high, low, close, 30)

def ADX_ext40n(high, low, close, n):
    mdi = talib.MINUS_DI(high, low, close)
    pdi = talib.PLUS_DI(high, low, close)
    adx = talib.ADX(high, low, close)
    tmp1 = adx -mdi
    for i in range(tmp1.shape[0] - n):
        idx = tmp1.shape[0]-1-i
        tmp1[idx] = tmp1[idx] - tmp1[idx-n]
    return tmp1
def ADX_ext401(high, low, close):
    return ADX_ext40n(high, low, close, 1)
def ADX_ext402(high, low, close):
    return ADX_ext40n(high, low, close, 2)
def ADX_ext403(high, low, close):
    return ADX_ext40n(high, low, close, 3)
def ADX_ext404(high, low, close):
    return ADX_ext40n(high, low, close, 4)
def ADX_ext405(high, low, close):
    return ADX_ext40n(high, low, close, 5)
def ADX_ext410(high, low, close):
    return ADX_ext40n(high, low, close, 10)
def ADX_ext430(high, low, close):
    return ADX_ext40n(high, low, close, 30)

def MACD_ext10n(close, n):
    macd,macdsignal,macdhist = talib.MACD(close)
    for i in range(macdhist.shape[0] - n):
        idx = macdhist.shape[0]-1-i
        macdhist[idx] = macdhist[idx] - macdhist[idx-n]    
    return macdhist
def MACD_ext101(close):
    return MACD_ext10n(close, 1)
def MACD_ext102(close):
    return MACD_ext10n(close, 2)
def MACD_ext103(close):
    return MACD_ext10n(close, 3)
def MACD_ext104(close):
    return MACD_ext10n(close, 4)
def MACD_ext105(close):
    return MACD_ext10n(close, 5)
def MACD_ext110(close):
    return MACD_ext10n(close, 10)
def MACD_ext200(close, n):
    macd,macdsignal,macdhist = talib.MACD(close)
    return macdsignal - macd

def MACD_ext30n(close, n):
    macd,macdsignal,macdhist = talib.MACD(close)
    for i in range(macdhist.shape[0] - n):
        idx = macdhist.shape[0]-1-i
        macd[idx] = macdhist[idx] - macdhist[idx-n]    
    return macd
def MACD_ext301(close):
    return MACD_ext30n(close, 1)
def MACD_ext302(close):
    return MACD_ext30n(close, 2)
def MACD_ext303(close):
    return MACD_ext30n(close, 3)
def MACD_ext304(close):
    return MACD_ext30n(close, 4)
def MACD_ext305(close):
    return MACD_ext30n(close, 5)
def EMA_1(close):
    return talib.EMA(close, 4)
def EMA_10n(close,n):
    ema = EMA_1(close)
    for i in range(ema.shape[0] - n):
        idx = ema.shape[0]-1-i
        ema[idx] = ema[idx] / ema[idx-n]       
    return ema
def EMA_101(close):
    return EMA_10n(close, 1)
def EMA_102(close):
    return EMA_10n(close, 2)
def EMA_103(close):
    return EMA_10n(close, 3)
def EMA_104(close):
    return EMA_10n(close, 4)
def EMA_105(close):
    return EMA_10n(close, 5)
def EMA_110(close):
    return EMA_10n(close, 10)
def EMA_130(close):
    return EMA_10n(close, 30)
def EMA_2(close):
    return talib.EMA(close, 9)
def EMA_20n(close,n):
    ema = EMA_2(close)
    for i in range(ema.shape[0] - n):
        idx = ema.shape[0]-1-i
        ema[idx] = ema[idx] / ema[idx-n]       
    return ema
def EMA_201(close):
    return EMA_20n(close, 1)
def EMA_202(close):
    return EMA_20n(close, 2)
def EMA_203(close):
    return EMA_20n(close, 3)
def EMA_204(close):
    return EMA_20n(close, 4)
def EMA_205(close):
    return EMA_20n(close, 5)
def EMA_210(close):
    return EMA_20n(close, 10)
def EMA_230(close):
    return EMA_20n(close, 30)
def EMA_3(close):
    return talib.EMA(close, 18)
def EMA_30n(close,n):
    ema = EMA_3(close)
    for i in range(ema.shape[0] - n):
        idx = ema.shape[0]-1-i
        ema[idx] = ema[idx] / ema[idx-n]       
    return ema
def EMA_301(close):
    return EMA_30n(close, 1)
def EMA_302(close):
    return EMA_30n(close, 2)
def EMA_303(close):
    return EMA_30n(close, 3)
def EMA_304(close):
    return EMA_30n(close, 4)
def EMA_305(close):
    return EMA_30n(close, 5)
def EMA_310(close):
    return EMA_30n(close, 10)
def EMA_330(close):
    return EMA_30n(close, 30)
def RSI(close):
    return talib.RSI(close)
def RSI_10n(close,n):
    rsi = RSI(close)
    for i in range(rsi.shape[0] - n):
        idx = rsi.shape[0]-1-i
        rsi[idx] = rsi[idx] - rsi[idx-n]
    return rsi
def RSI_101(close):
    return RSI_10n(close, 1)
def RSI_102(close):
    return RSI_10n(close, 2)
def RSI_103(close):
    return RSI_10n(close, 3)
def RSI_104(close):
    return RSI_10n(close, 4)
def RSI_105(close):
    return RSI_10n(close, 5)
def RSI_110(close):
    return RSI_10n(close, 10)
def RSI_130(close):
    return RSI_10n(close, 30)

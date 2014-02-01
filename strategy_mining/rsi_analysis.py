#-*-encoding:gbk-*-
'''
���ܣ�
1.���۸��������룬���rsi����
2.���۸��������룬���ָ�������ڵ�ƽ���۸�����
3.��ȡ��ָ��index�µ�ǰ��۸�䶯
4.����ָ����ֵ��������� 
'''
import sys
import os
import talib
import numpy
'''
������������ݣ�����talibrsi����������rsi
'''
def get_rsi(price_list, timewindow):
    size = len(price_list)
    if timewindow < 0 or size < timewindow:
        return None
    rsi_index = talib.RSI(price_list, timewindow)
    return rsi_index

'''
���ݹ�Ʊ�۸����ݣ���ȡƽ��ֵ����
'''
def get_avg_price(price_list, timewindow):
    size = len(price_list)
    if timewindow < 0 or size < timewindow:
        return None
    price_sum = talib.SUM(price_list, timewindow)
    return price_sum

'''
���������ƽ��ֵ������index���ж����ǻ��ǵ�
'''
def is_priceup(price_sum, index, threshold):
    if len(price_sum)<= index + 1 or index < 0:
        return None
    if price_sum[index + 1]/price_sum[index] > threshold + 1:
        return True
    elif price_sum[index + 1]/price_sum[index] < 1 - threshold:
        return False
    else:
        return None

'''
��������۸񣬻�ȡrsi/priceƽ��ֵ�����ж���ָ��index�µļ۸�仯��������rsi����һ�£����Ϊ1�� ����Ϊ0
'''
def get_sample(price_list):
    rsi_index = get_rsi(price_list, 14)
    price_sum = get_avg_price(price_list, 7)
    for s in range(0, len(rsi_index)):
        if rsi_index[s] == None:
            continue
        sign = None
        result = is_priceup(price_sum, s, 0.02)
        if None == result:
            continue
        if rsi_index[s] > 80:
            if False == result:
                sign = 0
            else:
                sign = 1
        elif rsi_index[s] < 15:
            if True == result:
                sign = 0
            else:
                sign = 1
        else:
            continue
        if sign == None:
            continue
        print "%.4f\t%.4f\t%.4f\t%d" %(rsi_index[s], price_sum[s], price_sum[s+1], sign)
    
'''
�����ļ�����ȡprice_list
'''        
def load_data(filename):
    tprice = []
    fd = open(filename, "r")
    for j in fd:
        line_list = j.rstrip().split(",")
        try:
            price = float(line_list[4])
        except Exception,e:
            continue
        tprice.append(price)
    price_list = numpy.array(tprice)
    return price_list

if __name__ == "__main__":
    price_list = load_data(sys.argv[1])
    get_sample(price_list)    

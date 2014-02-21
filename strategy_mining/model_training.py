#-*-encoding:gbk-*-

import sys,os
local_path = os.path.dirname(os.path.abspath(sys.argv[0]))
sys.path.append(local_path + "/./")

import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from price_judgement import *
from two_crow_builder import *
from three_inside_pattern import *
from three_inside_strike import *
import numpy
from sklearn import tree
from other_pattern import *
class base_model:
    #定义基本属性
    name = "base_model"
    #特征生成方法列表，其中，每一个特征的参数形式均为(prices_list, index,feature_result_list),其中，index表示该特征生成的下标，生成的结果会存放在feature_result_list下
    feature_builder_list = []
    sample_judgement = None
    model_predictor = None
    samples = []
    classes = []
    def __init__(self, feature_builder_list_input, sample_judgement_input, model_predictor_input):
        self.feature_builder_list = feature_builder_list_input
        self.sample_judgement = sample_judgement_input
        self.model_predictor = model_predictor_input

    def build_sample(self, open_price_list, high_price_list, low_price_list, close_price_list, adjust_close_list, volume_list, timewindow):
        for s in range(timewindow, len(open_price_list)-timewindow):
            prices = close_price_list[s-timewindow:s+timewindow]
            price_judge_high = high_price_list[s-timewindow:s]
            price_judge_low = low_price_list[s-timewindow:s]
            price_judge_open = open_price_list[s-timewindow:s]
            price_judge_close = close_price_list[s-timewindow:s]
            adjust_close = adjust_close_list[s-timewindow:s]
            volume = volume_list[s-timewindow:s]
            result = self.sample_judgement.judge(prices, 0.05)
            if result == None:
                continue
            result_list = []
            for m in range(0, len(self.feature_builder_list)):
                result_list.append(-1)

            for mindex, m in enumerate(feature_builder_list):
                m.feature_build(price_judge_open, price_judge_high, price_judge_low, price_judge_close, adjust_close, volume, mindex, result_list)
                #print result_list
            if result_list.count(0) == len(result_list):
                continue
            print result_list
            self.samples.append(result_list)
            self.classes.append(result)
            tmp_str = str(result)
            for s in result_list:
                tmp_str = tmp_str + "\t" + str(s)
            if result_list[0] != 0:
                print tmp_str

    def model_process(self):
        if len(self.samples) == 0:
            return
        model_predictor.fit(numpy.array(self.samples), numpy.array(self.classes))
        predict_value = model_predictor.predict_proba(numpy.array(self.samples))
        precision, recall, threshold = roc_curve(numpy.array(self.classes), predict_value[:,0])
        print precision
        print recall
        print threshold
        area = auc(recall, precision)
        print "auc = %.4f" %(area)

    def result_predict(self, open_price_list, high_price_list, low_price_list, close_price_list, adjust_close_list, volume_list, timewindow):
        samples = []
        
        open_price = open_price_list[-timewindow-1: -1]
        high_price = high_price_list[-timewindow-1: -1]
        low_price = low_price_list[-timewindow-1:-1]
        close_price = close_price_list[-timewindow - 1 : -1]
        adjust_price = adjust_close_list[-timewindow - 1 : -1]
        volume = volume_list[-timewindow - 1 : -1]
        for index, s in enumerate(self.feature_builder_list):
            samples.append(m.feature_build(price_judge_open, price_judge_high, price_judge_low, price_judge_close, adjust_close, volume, mindex, result_list))
        predict_value = self.model_predictor.predict(samples)
        return predict_value
 
    def result_predictprob(self, open_price_list, high_price_list, low_price_list, close_price_list, adjust_close_list, volume_list, timewindow):
        samples = []
        
        open_price = open_price_list[-timewindow-1: -1]
        high_price = high_price_list[-timewindow-1: -1]
        low_price = low_price_list[-timewindow-1:-1]
        close_price = close_price_list[-timewindow - 1 : -1]
        adjust_price = adjust_close_list[-timewindow - 1 : -1]
        volume = volume_list[-timewindow - 1 : -1]
        for index, s in enumerate(self.feature_builder_list):
            samples.append(m.feature_build(price_judge_open, price_judge_high, price_judge_low, price_judge_close, adjust_close, volume, mindex, result_list))
        predict_value = self.model_predictor.predict_prob(samples)
        return predict_value

def get_file_list():
    """hongbin0908@126.com
    a help function to load test data.
    """
    file_list = []
    for f in os.listdir(os.path.join(local_path, "../../tmp")):
        if f != None and not f.endswith(".csv"):
            continue
        file_list.append(os.path.join(local_path, "../../tmp", f))
         
    return file_list
if __name__ == "__main__":
    judger = prices_judgement()
    two_crow = twocrow_builder()
    three_inside = three_inside_up_builder()
    three_inside_strike = three_inside_strike_builder()
    three_outside_move = three_outside_move_builder()
    three_star_south = three_star_south_builder()
    three_ad_white_soldier = three_ad_white_soldier_builder()
    abandoned_baby = abandoned_baby_builder()
    three_ad_block = three_ad_block_builder()
    belt_hold = belt_hold_builder()
    break_away = breakaway_builder()
    conceal_baby = conceal_baby_swallow_builder()

    feature_builder_list = []
    feature_builder(talib.CDLCOUNTERATTACK, feature_builder_list)

    feature_builder(talib.CDLDARKCLOUDCOVER , feature_builder_list)
    feature_builder(talib.CDLDOJI , feature_builder_list)
    feature_builder(talib.CDLDOJISTAR , feature_builder_list)
    feature_builder(talib.CDLDRAGONFLYDOJI , feature_builder_list)
    feature_builder(talib.CDLENGULFING , feature_builder_list)
    feature_builder(talib.CDLEVENINGDOJISTAR , feature_builder_list)
    feature_builder(talib.CDLEVENINGSTAR , feature_builder_list)
    feature_builder(talib.CDLGAPSIDESIDEWHITE , feature_builder_list)

    feature_builder(talib.CDLGRAVESTONEDOJI  , feature_builder_list)
    feature_builder(talib.CDLHAMMER  , feature_builder_list)
    feature_builder(talib.CDLHANGINGMAN , feature_builder_list)
    feature_builder(talib.CDLHARAMI , feature_builder_list)
    feature_builder(talib.CDLHARAMICROSS , feature_builder_list)
    feature_builder(talib.CDLHIGHWAVE , feature_builder_list)
    feature_builder(talib.CDLHIKKAKE , feature_builder_list)
    feature_builder(talib.CDLHIKKAKEMOD , feature_builder_list)


    feature_builder(talib.CDLIDENTICAL3CROWS , feature_builder_list)
    feature_builder(talib.CDLINNECK , feature_builder_list)
    feature_builder(talib.CDLINVERTEDHAMMER , feature_builder_list)
    feature_builder(talib.CDLKICKING , feature_builder_list)
    feature_builder(talib.CDLKICKINGBYLENGTH , feature_builder_list)
    feature_builder(talib.CDLLADDERBOTTOM , feature_builder_list)
    feature_builder(talib.CDLLONGLEGGEDDOJI , feature_builder_list)
    feature_builder(talib.CDLLONGLINE , feature_builder_list)
    feature_builder(talib.CDLMARUBOZU , feature_builder_list)
    feature_builder(talib.CDLMATCHINGLOW , feature_builder_list)
    feature_builder(talib.CDLMATHOLD , feature_builder_list)
    feature_builder(talib.CDLMORNINGDOJISTAR , feature_builder_list)
    feature_builder(talib.CDLMORNINGSTAR , feature_builder_list)
    feature_builder(talib.CDLONNECK , feature_builder_list)
    feature_builder(talib.CDLPIERCING , feature_builder_list)

    feature_builder(talib.CDLRICKSHAWMAN , feature_builder_list)
    feature_builder(talib.CDLRISEFALL3METHODS , feature_builder_list)
    feature_builder(talib.CDLSEPARATINGLINES , feature_builder_list)
    feature_builder(talib.CDLSHOOTINGSTAR , feature_builder_list)
    feature_builder(talib.CDLSHORTLINE , feature_builder_list)
    feature_builder(talib.CDLSPINNINGTOP , feature_builder_list)
    feature_builder(talib.CDLSTALLEDPATTERN , feature_builder_list)

    feature_builder(talib.CDLSTICKSANDWICH , feature_builder_list)
    feature_builder(talib.CDLTAKURI , feature_builder_list)
    feature_builder(talib.CDLTASUKIGAP , feature_builder_list)
    feature_builder(talib.CDLTHRUSTING , feature_builder_list)
    feature_builder(talib.CDLTRISTAR , feature_builder_list)
    feature_builder(talib.CDLUNIQUE3RIVER , feature_builder_list)
    feature_builder(talib.CDLUPSIDEGAP2CROWS , feature_builder_list)

    feature_builder(talib.CDLXSIDEGAP3METHODS , feature_builder_list)

    feature_builder_list.append(two_crow)
    feature_builder_list.append(three_inside)
    feature_builder_list.append(three_inside_strike)
    feature_builder_list.append(three_outside_move)
    feature_builder_list.append(three_star_south)
    feature_builder_list.append(three_ad_white_soldier)
    feature_builder_list.append(abandoned_baby)
    feature_builder_list.append(three_ad_block)
    feature_builder_list.append(belt_hold)
    feature_builder_list.append(break_away)
    feature_builder_list.append(conceal_baby)
    print len(feature_builder_list)
    model_predictor = tree.DecisionTreeClassifier()
    model = base_model(feature_builder_list, judger, model_predictor)
    open_prices = []
    high_prices = []
    low_prices = []
    close_prices = []
    adjust_close = []
    volume = []
    file_list = ['tmp/A-2010-yahoofinance.csv','tmp/AA-2010-yahoofinance.csv','tmp/AAPL-2010-yahoofinance.csv','tmp/ABC-2010-yahoofinance.csv','tmp/ABT-2010-yahoofinance.csv','tmp/ACE-2010-yahoofinance.csv','tmp/ACN-2010-yahoofinance.csv','tmp/ACT-2010-yahoofinance.csv','tmp/ADBE-2010-yahoofinance.csv','tmp/ADI-2010-yahoofinance.csv','tmp/ADM-2010-yahoofinance.csv','tmp/ADP-2010-yahoofinance.csv','tmp/ADS-2010-yahoofinance.csv','tmp/ADSK-2010-yahoofinance.csv','tmp/AEE-2010-yahoofinance.csv','tmp/AEP-2010-yahoofinance.csv','tmp/AES-2010-yahoofinance.csv','tmp/AET-2010-yahoofinance.csv','tmp/AFL-2010-yahoofinance.csv','tmp/AGN-2010-yahoofinance.csv','tmp/AIG-2010-yahoofinance.csv','tmp/AIV-2010-yahoofinance.csv','tmp/AIZ-2010-yahoofinance.csv','tmp/AKAM-2010-yahoofinance.csv','tmp/ALL-2010-yahoofinance.csv','tmp/ALTR-2010-yahoofinance.csv','tmp/ALXN-2010-yahoofinance.csv','tmp/AMAT-2010-yahoofinance.csv','tmp/AME-2010-yahoofinance.csv','tmp/AMGN-2010-yahoofinance.csv','tmp/AMP-2010-yahoofinance.csv','tmp/AMT-2010-yahoofinance.csv','tmp/AMZN-2010-yahoofinance.csv','tmp/AN-2010-yahoofinance.csv','tmp/AON-2010-yahoofinance.csv','tmp/APA-2010-yahoofinance.csv','tmp/APC-2010-yahoofinance.csv','tmp/APD-2010-yahoofinance.csv','tmp/APH-2010-yahoofinance.csv','tmp/ARG-2010-yahoofinance.csv','tmp/ATI-2010-yahoofinance.csv','tmp/AVB-2010-yahoofinance.csv','tmp/AVP-2010-yahoofinance.csv','tmp/AVY-2010-yahoofinance.csv','tmp/AXP-2010-yahoofinance.csv','tmp/AZO-2010-yahoofinance.csv','tmp/BA-2010-yahoofinance.csv','tmp/BAC-2010-yahoofinance.csv','tmp/BAX-2010-yahoofinance.csv','tmp/BBBY-2010-yahoofinance.csv','tmp/BBT-2010-yahoofinance.csv','tmp/BBY-2010-yahoofinance.csv','tmp/BCR-2010-yahoofinance.csv','tmp/BDX-2010-yahoofinance.csv','tmp/BEAM-2010-yahoofinance.csv','tmp/BEN-2010-yahoofinance.csv','tmp/BF-B-2010-yahoofinance.csv','tmp/BHI-2010-yahoofinance.csv','tmp/BIIB-2010-yahoofinance.csv','tmp/BK-2010-yahoofinance.csv','tmp/BLK-2010-yahoofinance.csv','tmp/BLL-2010-yahoofinance.csv','tmp/BMS-2010-yahoofinance.csv','tmp/BMY-2010-yahoofinance.csv','tmp/BRCM-2010-yahoofinance.csv','tmp/BRK-B-2010-yahoofinance.csv','tmp/BSX-2010-yahoofinance.csv','tmp/BTU-2010-yahoofinance.csv','tmp/BXP-2010-yahoofinance.csv','tmp/C-2010-yahoofinance.csv','tmp/CA-2010-yahoofinance.csv','tmp/CAG-2010-yahoofinance.csv','tmp/CAH-2010-yahoofinance.csv','tmp/CAM-2010-yahoofinance.csv','tmp/CAT-2010-yahoofinance.csv','tmp/CB-2010-yahoofinance.csv','tmp/CBG-2010-yahoofinance.csv','tmp/CBS-2010-yahoofinance.csv','tmp/CCE-2010-yahoofinance.csv','tmp/CCI-2010-yahoofinance.csv','tmp/CCL-2010-yahoofinance.csv','tmp/CELG-2010-yahoofinance.csv','tmp/CERN-2010-yahoofinance.csv','tmp/CF-2010-yahoofinance.csv','tmp/CFN-2010-yahoofinance.csv','tmp/CHK-2010-yahoofinance.csv','tmp/CHRW-2010-yahoofinance.csv','tmp/CI-2010-yahoofinance.csv','tmp/CINF-2010-yahoofinance.csv','tmp/CL-2010-yahoofinance.csv','tmp/CLF-2010-yahoofinance.csv','tmp/CLX-2010-yahoofinance.csv','tmp/CMA-2010-yahoofinance.csv','tmp/CMCSA-2010-yahoofinance.csv','tmp/CME-2010-yahoofinance.csv','tmp/CMG-2010-yahoofinance.csv','tmp/CMI-2010-yahoofinance.csv','tmp/CMS-2010-yahoofinance.csv','tmp/CNP-2010-yahoofinance.csv','tmp/CNX-2010-yahoofinance.csv','tmp/COF-2010-yahoofinance.csv','tmp/COG-2010-yahoofinance.csv','tmp/COH-2010-yahoofinance.csv','tmp/COL-2010-yahoofinance.csv','tmp/COP-2010-yahoofinance.csv','tmp/COST-2010-yahoofinance.csv','tmp/COV-2010-yahoofinance.csv','tmp/CPB-2010-yahoofinance.csv','tmp/CRM-2010-yahoofinance.csv','tmp/CSC-2010-yahoofinance.csv','tmp/CSCO-2010-yahoofinance.csv','tmp/CSX-2010-yahoofinance.csv','tmp/CTAS-2010-yahoofinance.csv','tmp/CTL-2010-yahoofinance.csv','tmp/CTSH-2010-yahoofinance.csv','tmp/CTXS-2010-yahoofinance.csv','tmp/CVC-2010-yahoofinance.csv','tmp/CVS-2010-yahoofinance.csv','tmp/CVX-2010-yahoofinance.csv','tmp/D-2010-yahoofinance.csv','tmp/DAL-2010-yahoofinance.csv','tmp/DD-2010-yahoofinance.csv','tmp/DE-2010-yahoofinance.csv','tmp/DFS-2010-yahoofinance.csv','tmp/DG-2010-yahoofinance.csv','tmp/DGX-2010-yahoofinance.csv','tmp/DHI-2010-yahoofinance.csv','tmp/DHR-2010-yahoofinance.csv','tmp/DIS-2010-yahoofinance.csv','tmp/DISCA-2010-yahoofinance.csv','tmp/DLTR-2010-yahoofinance.csv','tmp/DNB-2010-yahoofinance.csv','tmp/DNR-2010-yahoofinance.csv','tmp/DO-2010-yahoofinance.csv','tmp/DOV-2010-yahoofinance.csv','tmp/DOW-2010-yahoofinance.csv','tmp/DPS-2010-yahoofinance.csv','tmp/DRI-2010-yahoofinance.csv','tmp/DTE-2010-yahoofinance.csv','tmp/DTV-2010-yahoofinance.csv','tmp/DUK-2010-yahoofinance.csv','tmp/DVA-2010-yahoofinance.csv','tmp/DVN-2010-yahoofinance.csv','tmp/EA-2010-yahoofinance.csv','tmp/EBAY-2010-yahoofinance.csv','tmp/ECL-2010-yahoofinance.csv','tmp/ED-2010-yahoofinance.csv','tmp/EFX-2010-yahoofinance.csv','tmp/EIX-2010-yahoofinance.csv','tmp/EL-2010-yahoofinance.csv','tmp/EMC-2010-yahoofinance.csv','tmp/EMN-2010-yahoofinance.csv','tmp/EMR-2010-yahoofinance.csv','tmp/EOG-2010-yahoofinance.csv','tmp/EQR-2010-yahoofinance.csv','tmp/EQT-2010-yahoofinance.csv','tmp/ESRX-2010-yahoofinance.csv','tmp/ESV-2010-yahoofinance.csv','tmp/ETFC-2010-yahoofinance.csv','tmp/ETN-2010-yahoofinance.csv','tmp/ETR-2010-yahoofinance.csv','tmp/EW-2010-yahoofinance.csv','tmp/EXC-2010-yahoofinance.csv','tmp/EXPD-2010-yahoofinance.csv','tmp/EXPE-2010-yahoofinance.csv','tmp/F-2010-yahoofinance.csv','tmp/FAST-2010-yahoofinance.csv','tmp/FCX-2010-yahoofinance.csv','tmp/FDO-2010-yahoofinance.csv','tmp/FDX-2010-yahoofinance.csv','tmp/FE-2010-yahoofinance.csv','tmp/FFIV-2010-yahoofinance.csv','tmp/FIS-2010-yahoofinance.csv','tmp/FISV-2010-yahoofinance.csv','tmp/FITB-2010-yahoofinance.csv','tmp/FLIR-2010-yahoofinance.csv','tmp/FLR-2010-yahoofinance.csv','tmp/FLS-2010-yahoofinance.csv','tmp/FMC-2010-yahoofinance.csv','tmp/FOSL-2010-yahoofinance.csv','tmp/FOXA-2010-yahoofinance.csv','tmp/FRX-2010-yahoofinance.csv','tmp/FSLR-2010-yahoofinance.csv','tmp/FTI-2010-yahoofinance.csv','tmp/FTR-2010-yahoofinance.csv','tmp/GAS-2010-yahoofinance.csv','tmp/GCI-2010-yahoofinance.csv','tmp/GD-2010-yahoofinance.csv','tmp/GE-2010-yahoofinance.csv','tmp/GHC-2010-yahoofinance.csv','tmp/GILD-2010-yahoofinance.csv','tmp/GIS-2010-yahoofinance.csv','tmp/GLW-2010-yahoofinance.csv','tmp/GME-2010-yahoofinance.csv','tmp/GNW-2010-yahoofinance.csv','tmp/GOOG-2010-yahoofinance.csv','tmp/GPC-2010-yahoofinance.csv','tmp/GPS-2010-yahoofinance.csv','tmp/GRMN-2010-yahoofinance.csv','tmp/GS-2010-yahoofinance.csv','tmp/GT-2010-yahoofinance.csv','tmp/GWW-2010-yahoofinance.csv','tmp/HAL-2010-yahoofinance.csv','tmp/HAR-2010-yahoofinance.csv','tmp/HAS-2010-yahoofinance.csv','tmp/HBAN-2010-yahoofinance.csv','tmp/HCBK-2010-yahoofinance.csv','tmp/HCN-2010-yahoofinance.csv','tmp/HCP-2010-yahoofinance.csv','tmp/HD-2010-yahoofinance.csv','tmp/HES-2010-yahoofinance.csv','tmp/HIG-2010-yahoofinance.csv','tmp/HOG-2010-yahoofinance.csv','tmp/HON-2010-yahoofinance.csv','tmp/HOT-2010-yahoofinance.csv','tmp/HP-2010-yahoofinance.csv','tmp/HPQ-2010-yahoofinance.csv','tmp/HRB-2010-yahoofinance.csv','tmp/HRL-2010-yahoofinance.csv','tmp/HRS-2010-yahoofinance.csv','tmp/HSP-2010-yahoofinance.csv','tmp/HST-2010-yahoofinance.csv','tmp/HSY-2010-yahoofinance.csv','tmp/HUM-2010-yahoofinance.csv','tmp/IBM-2010-yahoofinance.csv','tmp/ICE-2010-yahoofinance.csv','tmp/IFF-2010-yahoofinance.csv','tmp/IGT-2010-yahoofinance.csv','tmp/INTC-2010-yahoofinance.csv','tmp/INTU-2010-yahoofinance.csv','tmp/IP-2010-yahoofinance.csv','tmp/IPG-2010-yahoofinance.csv','tmp/IR-2010-yahoofinance.csv','tmp/IRM-2010-yahoofinance.csv','tmp/ISRG-2010-yahoofinance.csv','tmp/ITW-2010-yahoofinance.csv','tmp/IVZ-2010-yahoofinance.csv','tmp/JBL-2010-yahoofinance.csv','tmp/JCI-2010-yahoofinance.csv','tmp/JEC-2010-yahoofinance.csv','tmp/JNJ-2010-yahoofinance.csv','tmp/JNPR-2010-yahoofinance.csv','tmp/JOY-2010-yahoofinance.csv','tmp/JPM-2010-yahoofinance.csv','tmp/JWN-2010-yahoofinance.csv','tmp/K-2010-yahoofinance.csv','tmp/KEY-2010-yahoofinance.csv','tmp/TWX-2010-yahoofinance.csv','tmp/TXN-2010-yahoofinance.csv','tmp/TXT-2010-yahoofinance.csv','tmp/TYC-2010-yahoofinance.csv','tmp/UNH-2010-yahoofinance.csv','tmp/UNM-2010-yahoofinance.csv','tmp/UNP-2010-yahoofinance.csv','tmp/UPS-2010-yahoofinance.csv','tmp/URBN-2010-yahoofinance.csv','tmp/USB-2010-yahoofinance.csv','tmp/UTX-2010-yahoofinance.csv','tmp/V-2010-yahoofinance.csv','tmp/VAR-2010-yahoofinance.csv','tmp/VFC-2010-yahoofinance.csv','tmp/VIAB-2010-yahoofinance.csv','tmp/VLO-2010-yahoofinance.csv','tmp/VMC-2010-yahoofinance.csv','tmp/VNO-2010-yahoofinance.csv','tmp/VRSN-2010-yahoofinance.csv','tmp/VRTX-2010-yahoofinance.csv','tmp/VTR-2010-yahoofinance.csv','tmp/VZ-2010-yahoofinance.csv','tmp/WAG-2010-yahoofinance.csv','tmp/WAT-2010-yahoofinance.csv','tmp/WDC-2010-yahoofinance.csv','tmp/WEC-2010-yahoofinance.csv','tmp/WFC-2010-yahoofinance.csv','tmp/WFM-2010-yahoofinance.csv','tmp/WHR-2010-yahoofinance.csv','tmp/WIN-2010-yahoofinance.csv','tmp/WLP-2010-yahoofinance.csv','tmp/WM-2010-yahoofinance.csv','tmp/WMB-2010-yahoofinance.csv','tmp/WMT-2010-yahoofinance.csv','tmp/WU-2010-yahoofinance.csv','tmp/WY-2010-yahoofinance.csv','tmp/WYN-2010-yahoofinance.csv','tmp/WYNN-2010-yahoofinance.csv','tmp/X-2010-yahoofinance.csv','tmp/XEL-2010-yahoofinance.csv','tmp/XL-2010-yahoofinance.csv','tmp/XLNX-2010-yahoofinance.csv','tmp/XOM-2010-yahoofinance.csv','tmp/XRAY-2010-yahoofinance.csv','tmp/XRX-2010-yahoofinance.csv','tmp/YHOO-2010-yahoofinance.csv','tmp/YUM-2010-yahoofinance.csv','tmp/ZION-2010-yahoofinance.csv','tmp/ZMH-2010-yahoofinance.csv']
    file_list = get_file_list()
    for s in file_list:
        open_prices = []
        high_prices = []
        low_prices = []
        close_prices = []
        adjust_close = []
        volume = []
        load_data(s, open_prices, high_prices, low_prices, close_prices, adjust_close, volume)
        model.build_sample(numpy.array(open_prices), numpy.array(high_prices), numpy.array(low_prices), numpy.array(close_prices), numpy.array(adjust_close), numpy.array(volume), 7)
   
    model.model_process()
#    input_test = numpy.array([-100, 100])
#    print model.result_predict(input_test)

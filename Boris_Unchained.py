import datetime, time
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from pandas_datareader import data as pdr
from collections import deque
import os
import alpaca_trade_api as api
from math import floor
###################################
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import yfinance as yfin
from googlefinance.client import get_price_data, get_prices_data, get_prices_time_data


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist

import time
import datetime
# #
# t = datetime.datetime.today()
# print(t)
# future = datetime.datetime(t.year,t.month,t.day,7,30)
# print(future)
#
# time.sleep((future-t).total_seconds())

alpaca = api.REST('AKW8P9R6JZE3JIGBYX4X', 'UEDTXjYR8b0z2ogMSCVb4oWVHpMZyJTMMxtsJlmc', 'https://api.alpaca.markets')


while True:

    # for dirname, _, filenames in os.walk('/kaggle/input'):
    #     for filename in filenames:
    #         print(os.path.join(dirname, filename))


    param = {
    	'q': ".DJI", # Stock symbol (ex: "AAPL")
    	'i': "86400", # Interval size in seconds ("86400" = 1 day intervals)
    	'x': "INDEXDJX", # Stock exchange symbol on which stock is traded (ex: "NASD")
    	'p': "1Y" # Period (Ex: "1Y" = 1 year)
    }

    df = get_price_data(param)
    print(df)

    params = [
    	# Dow Jones
    	{
    		'q': ".DJI",
    		'x': "INDEXDJX",
    	},
    	# NYSE COMPOSITE (DJ)
    	{
    		'q': "NYA",
    		'x': "INDEXNYSEGIS",
    	},
    	# S&P 500
    	{
    		'q': ".INX",
    		'x': "INDEXSP",
    	}
    ]
    period = "1Y"
    df = get_prices_data(params, period)
    print(df)

    params = [
    	# Dow Jones
    	{
    		'q': ".DJI",
    		'x': "INDEXDJX",
    	},
    	# NYSE COMPOSITE (DJ)
    	{
    		'q': "NYA",
    		'x': "INDEXNYSEGIS",
    	},
    	# S&P 500
    	{
    		'q': ".INX",
    		'x': "INDEXSP",
    	}
    ]
    period = "1Y"
    interval = 60*30 # 30 minutes

    df = get_prices_time_data(params, period, interval)
    print(df)
    df


    from pandas_datareader import data as pdr

    from collections import deque

    class PSAR:

      def __init__(self, init_af=0.02, max_af=0.2, af_step=0.02):
        self.max_af = max_af
        self.init_af = init_af
        self.af = init_af
        self.af_step = af_step
        self.extreme_point = None
        self.high_price_trend = []
        self.low_price_trend = []
        self.high_price_window = deque(maxlen=2)
        self.low_price_window = deque(maxlen=2)

        # Lists to track results
        self.psar_list = []
        self.af_list = []
        self.ep_list = []
        self.high_list = []
        self.low_list = []
        self.trend_list = []
        self._num_days = 0

      def calcPSAR(self, high, low):
        if self._num_days >= 3:
          psar = self._calcPSAR()
        else:
          psar = self._initPSARVals(high, low)

        psar = self._updateCurrentVals(psar, high, low)
        self._num_days += 1

        return psar

      def _initPSARVals(self, high, low):
        if len(self.low_price_window) <= 1:
          self.trend = None
          self.extreme_point = high
          return None

        if self.high_price_window[0] < self.high_price_window[1]:
          self.trend = 1
          psar = min(self.low_price_window)
          self.extreme_point = max(self.high_price_window)
        else:
          self.trend = 0
          psar = max(self.high_price_window)
          self.extreme_point = min(self.low_price_window)

        return psar

      def _calcPSAR(self):
        prev_psar = self.psar_list[-1]
        if self.trend == 1: # Up
          psar = prev_psar + self.af * (self.extreme_point - prev_psar)
          psar = min(psar, min(self.low_price_window))
        else:
          psar = prev_psar - self.af * (prev_psar - self.extreme_point)
          psar = max(psar, max(self.high_price_window))

        return psar

      def _updateCurrentVals(self, psar, high, low):
        if self.trend == 1:
          self.high_price_trend.append(high)
        elif self.trend == 0:
          self.low_price_trend.append(low)

        psar = self._trendReversal(psar, high, low)

        self.psar_list.append(psar)
        self.af_list.append(self.af)
        self.ep_list.append(self.extreme_point)
        self.high_list.append(high)
        self.low_list.append(low)
        self.high_price_window.append(high)
        self.low_price_window.append(low)
        self.trend_list.append(self.trend)

        return psar

      def _trendReversal(self, psar, high, low):
        # Checks for reversals
        reversal = False
        if self.trend == 1 and psar > low:
          self.trend = 0
          psar = max(self.high_price_trend)
          self.extreme_point = low
          reversal = True
        elif self.trend == 0 and psar < high:
          self.trend = 1
          psar = min(self.low_price_trend)
          self.extreme_point = high
          reversal = True

        if reversal:
          self.af = self.init_af
          self.high_price_trend.clear()
          self.low_price_trend.clear()
        else:
            if high > self.extreme_point and self.trend == 1:
              self.af = min(self.af + self.af_step, self.max_af)
              self.extreme_point = high
            elif low < self.extreme_point and self.trend == 0:
              self.af = min(self.af + self.af_step, self.max_af)
              self.extreme_point = low

        return psar

    yfin.pdr_override()

    # Stock_List = ['GDX','NRGU','TQQQ','SQQQ','SPXU','HIBL','FNGU','FNGD','NUGT','DUST','FAS','FAZ','LABU','LABD','SOXL','SOXS','CURE','TMF','TMV','DFEN','DRV','DRN','SPY','QQQ','DIA','XLK','SOXX','XLE','XOP','XLY','IYR','XLB','XLU','XLP','XLF','SH','GDX','EFA']
    Stock_List = ['SMCI','SPY','GDX']


    # ,'HYG','TLT','PCY'
    # Stock_List = ['SPY','QQQ','DIA','XLK','SOXX','XLE','XOP','XLY','IYR','XLB','XLU','XLP','XLF','SH','SQQQ','GDX','EFA','HYG','TLT','PCY']


    Stock_Name = Stock_List[0]
    aapl = pdr.get_data_yahoo(Stock_Name,
                start=datetime.date.today() + datetime.timedelta(-50),
                end=datetime.date.today(), interval="15m")

    length = len(aapl)

    CS_ = []
    Cash_ = []
    Stock_Change = [0]*length
    Stock_ = [[0]*length for _ in Stock_List]
    Stock_Exchange = []
    Stocks_Change = []
    Stock_Names = []
    Monthly = []
    Weekly = []
    FishIndicator = 0
    Fish1_ = []
    Fish2_ = []
    Fish3_ = []

    CST = 0

    for b in range(2):
        for ii in range(len(Stock_List)):
            Stock_Name = Stock_List[ii]
            aaplvv = 0
            aaploo = 0
            aaplcc = 0
            aaplll = 0
            aaplhh = 0

            for o in range(1):
                if o == 0:
                    interval = "15m"
                    delta = -59
                    FishI = Fish1_
                if o == 2:
                    interval = "1h"
                    delta = -59
                    FishI = Fish2_
                if o == 1:
                    interval = "1d"
                    delta = -1000
                    FishI = Fish3_
                # if o == 1:
                #     Stock_Name = Stock_Name[0]
                print(o)
                if o == 0:
                    yfObj = yfin.Ticker(Stock_Name)
                    data = yfObj.history(period='5d', interval=interval)
            # #         data = yfObj.history(start='2007-01-01', end='2009-01-01', interval=interval)
                    #data = yfObj.history(start='2017-01-02', end='2019-01-01', interval=interval)

                    indic = PSAR()
                    aapl = pdr.get_data_yahoo(Stock_Name,period='5d', interval=interval)

                    length = len(aapl)

                    aaplv = aapl.iloc[0:length,4]
                    aaplc = aapl.iloc[0:length,3]
                    aapll = aapl.iloc[0:length,2]
                    aaplh = aapl.iloc[0:length,1]
                    aaplo = aapl.iloc[0:length,0]
                    aaplv = aaplv.values.tolist()
                    aaplo = aaplo.values.tolist()
                    aaplc = aaplc.values.tolist ()
                    aapll = aapll.values.tolist()
                    aaplh = aaplh.values.tolist ()

                    aaplvv = aaplv[len(aaplv)-2]
                    aaploo = aaplo[len(aaplo)-2]
                    aaplcc = aaplc[len(aaplc)-2]
                    aaplll = aapll[len(aapll)-2]
                    aaplhh = aaplh[len(aaplh)-2]

                    o = 1

                    interval = "1d"
                    delta = -1000
                    FishI = Fish3_
                    Stock_Name = Stock_List[ii]

                if o == 1:
                    yfObj = yfin.Ticker(Stock_Name)
                    data = yfObj.history(start=(datetime.date.today() + datetime.timedelta(delta)), end=datetime.date.today(), interval=interval)
            # #         data = yfObj.history(start='2007-01-01', end='2009-01-01', interval=interval)
                    #data = yfObj.history(start='2017-01-02', end='2019-01-01', interval=interval)


                    indic = PSAR()
                    aapl = pdr.get_data_yahoo(Stock_Name,
                                          start=datetime.date.today() + datetime.timedelta(delta),
                                          end=datetime.date.today(), interval=interval)

        #         aapl = pdr.get_data_yahoo(Stock_Name,start='2007-01-01', end='2009-01-01', interval=interval)
         #       aapl = pdr.get_data_yahoo(Stock_Name, start='2017-01-02', end='2019-01-01', interval=interval)

                length = len(aapl)

                aaplv = aapl.iloc[0:length,4]
                aaplc = aapl.iloc[0:length,3]
                aapll = aapl.iloc[0:length,2]
                aaplh = aapl.iloc[0:length,1]
                aaplo = aapl.iloc[0:length,0]
                aaplv = aaplv.values.tolist()
                aaplo = aaplo.values.tolist()
                aaplc = aaplc.values.tolist ()
                aapll = aapll.values.tolist()
                aaplh = aaplh.values.tolist ()

                if o == 1:
                    aaplv.append(aaplvv)
                    aaplo.append(aaploo)
                    aaplc.append(aaplcc)
                    aapll.append(aaplll)
                    aaplh.append(aaplhh)
                    print(aaplcc)
        #             length = len(aapl)

            #     print(len(aaplc))
                # aaplo
                aapl['Log returns'] = np.log(aapl['Close']/aapl['Close'].shift())
                aapl['Log returns'].std()
                volatility = aapl['Log returns'].std()*252**.5

                minneg = []
                maxpos = []
                maxhigh = []
                minlow = []
                avgpos = 0
                avgneg = 0
                OBV = 0
                N = 0
                SMMA = 0
                Fish = 0
                OBVindicator = 0
                PeriodHigh = 0
                PeriodLow = 0
                diff_ = []
                Value3 = 0


                length_time = 14


                for i in range(length_time):
                    try:
                        N = ((aaplc[i]-aapll[i])-(aaplh[i]-aaplc[i]))/(aaplh[i]-aapll[i])
                    except:
                        try:
                            N = ((aaplc[i-1]-aapll[i-1])-(aaplh[i-1]-aaplc[i-1]))/(aaplh[i-1]-aapll[i-1])
                        except:
                            N = 0.5

                        SMMA += aaplc[i]

                    if aaplo[i]-aaplc[i] > 0:
                        pos = aaplo[i]-aaplc[i]
                        avgpos += pos
                        OBVpre = OBV
                        try:
                            OBV += (aaplv[i]/aaplv[i-1])/(i+1)
                        except:
                            try:
                                OBV += (aaplv[i-1]/aaplv[i-2])/i
                            except:
                                OBV += (aaplv[i-2]/aaplv[i-3])/i

                    else:
                        neg = aaplo[i]-aaplc[i]
                        avgneg += neg
                        OBVpre = OBV
                        try:
                            OBV -= (aaplv[i]/aaplv[i-1])
                        except:
                            try:
                                OBV -= (aaplv[i-1]/aaplv[i-2])
                            except:
                                OBV -= (aaplv[i-2]/aaplv[i-3])


                    maxpos.append(aaplc[i])
                    maxhigh.append(aaplh[i])
                    minlow.append(aapll[i])
                    OBVindicator = abs(OBVpre)-abs(OBV)

                minneg = min(maxpos)
                maxposv = max(maxpos)

                maxhigh = max(maxhigh)
                minlow = min(minlow)

                for i in range(len(maxpos)):
                    if minneg == maxpos[i]:
                        PeriodHigh = i
                    if maxpos == maxpos[i]:
                        PeriodLow = i

                AroonUp = (-(PeriodHigh - length_time)/30)*100
                AroonDown = ((PeriodLow - length_time)/30)*100
                Aroon = (AroonUp + AroonDown)
                # maxposv = max(maxpos)
                avgneg = avgneg/length_time
                avgpos = avgpos/length_time


        #         for i in range(length_time):
        #             Price = (aaplh[i] + aapll[i]) / 2
        #             Value3 = max(-0.9999, min(0.9999, 0.5 * 2 * ((Price - minlow) / (maxhigh - minlow) - 0.5) + 0.5 *  Value3))
        #             Fish = 0.25 * math.log((1 + Value3) / (1 - Value3)) + 0.5 * Fish

                # print(avgneg)
                # print(avgpos)
                SMMA = SMMA/length_time
                K = ((aaplc[length_time] - minneg)/(maxposv - minneg)) *100
                RSI = 100 - (100/(1+(avgpos/(-avgneg))))

                # print(K)
                # print(RSI)
                # print(OBV)
                # print(N)
                # print(SMMA)
                # print(AroonUp)
                # print(AroonDown)
                #Chaikin Indicator

                import numpy             as np
                import matplotlib.pyplot as plt
                import matplotlib
                import collections

                account_dict = collections.defaultdict(list)
                account_dict1 = collections.defaultdict(list)

                value = 0
                Cash = 50000
                Stock = 50000
                timestamp = 182
                Rule_1_output = 0
                Rule_2_output = 0
                Rule_3_output = 0
                Rule_4_output = 0
                Rule_1S_output = 0
                Rule_2S_output = 0
                Rule_3S_output = 0
                Rule_4S_output = 0

                #Max Weight = 5

                weight1 = 1
                weight2 = 1
                weight3 = 1
                weight4 = 1
                weight5 = 1
                weight6 = 1
                weight7 = 1

                diff = 0
                diffW = 0
                diffM = 0
                # weight1_ = []
                # weight2_ = []
                # weight3_ = []
                # weight4_ = []
                Value = 0.75
                Buy = []
                Sell = []
                stock = aaplc
                day = []
                diff_ = []
                diff_W_ = []
                Track_Change = []
                Stock = []

                K_ = []
                RSI_ = []
                OBV_ = []
                N_ = []
                SMMA_ = []
                Aroon_ = []
                MACD_ = []
                Fish_ = []
                Value3_ = []

                K__ = []
                OBV__ = []
                RSI__ = []
                N__ = []
                MACD__ = []
                Fish__ = []
                Value3 = 1
                def Indicator(value1, value2, Fish, Value3):
                    minneg = []
                    maxpos = []
                    avgpos = 0
                    avgneg = 0
                    maxhigh = []
                    minlow = []
                    OBV = 0
                    N = 0
                    SMMA = 0
                    Fish = 0
                    OBVindicator = 0
                    Williams = 0
                    EMA = 0
                    EMA12 = 0
                    EMA26 = 0
                    MACD = 0
                    Signal_Line = 0


                    for i in range((value1-value2),value1):
                        Signal_Line = 0

                        #i = 1831
                        #################################
                        for l in range((value1-9),value1):

                            for j in range((value1-12),value1):
                                EMA += aaplc[j]
                                # print(EMA)
                            EMA12 = EMA/12
                            EMA12 += ((aaplc[j] - EMA12)  * (2/12)) + EMA12
                            EMA = 0

                            for k in range((value1-26),value1):
                                EMA += aaplc[k]

                            EMA26 = EMA/26
                            EMA26 += ((aaplc[k] - EMA26) * (2/26)) + EMA26

                            MACD = EMA12 - EMA26
                            Signal_Line += MACD

                        Signal_Line = Signal_Line/9

                        # print()
                        # print(MACD)
                        # print(Signal_Line)

                        if Signal_Line < MACD:
                            MACD = 1
                        if Signal_Line > MACD:
                            MACD = 0
                        else:
                            MACD = 1

                        # print(MACD)
                        #################################
                        PeriodHigh = 0
                        PeriodLow = 0
                        try:
                            N = ((aaplc[i]-aapll[i])-(aaplh[i]-aaplc[i]))/(aaplh[i]-aapll[i])
                        except:
                            try:
                                N = ((aaplc[i-1]-aapll[i-1])-(aaplh[i-1]-aaplc[i-1]))/(aaplh[i-1]-aapll[i-1])
                            except:
                                try:
                                    N = ((aaplc[i-2]-aapll[i-2])-(aaplh[i-2]-aaplc[i-2]))/(aaplh[i-2]-aapll[i-2])
                                except:
                                    N = 0.5

                        SMMA += aaplc[i]
                        if aaplo[i]-aaplc[i] > 0:
                            pos = aaplo[i]-aaplc[i]
                            avgpos += pos
                            OBVpre = OBV
                            try:
                                OBV += (aaplv[i]/aaplv[i-1])/(i+1)
                            except:
                                OBV += (aaplv[i-1]/aaplv[i-2])
                        else:
                            neg = aaplo[i]-aaplc[i]
                            avgneg += neg
                            OBVpre = OBV
                            try:
                                OBV -= (aaplv[i]/aaplv[i-1])
                            except:
                                try:
                                    OBV -= (aaplv[i-1]/aaplv[i-2])
                                except:
                                    OBV -= (aaplv[i-2]/aaplv[i-3])
                        maxpos.append(aaplc[i])
                        OBVindicator = abs(OBVpre)-abs(OBV)

                        maxhigh.append(aaplh[i])
                        minlow.append(aapll[i])

                    minneg = min(maxpos)
                    maxposv = max(maxpos)

                    maxhigh = max(maxhigh)
                    minlow = min(minlow)


                    for i in range(len(maxpos)):
                        if minneg == maxpos[i]:
                            PeriodHigh = i
                        if maxpos == maxpos[i]:
                            PeriodLow = i
            #         Williams = (PeriodHigh-aaplc[i])/(PeriodHigh-PeriodLow)
                    AroonUp = (-(PeriodHigh - length_time)/30)*100
                    AroonDown = ((PeriodLow - length_time)/30)*100
                    Aroon = (AroonUp + AroonDown)

                    import math
                    if value2 == 14:
                        Value3 = 0
                        Fish = 0
                    for i in range((value1-value2),value1):
                        Price = (aaplh[i] + aapll[i]) / 2
                        Value3 = max(-0.9999, (min(0.9999, (((Price - minlow) / (maxhigh - minlow) - 0.6) + 0.6 *  Value3))))
                        Fish = 0.25 * math.log((1 + Value3) / (1 - Value3)) + 0.5 * Fish
            #             Value3 = -Value3*2
            #             Fish = -Fish*2


            #             import numpy as np3
            #             from sklearn.preprocessing import normalize


            #             x = np.random.rand(1000)*10
            #             norm1 = x / np.linalg.norm(x)
            #             norm2 = normalize(x[:,np.newaxis], axis=0).ravel()
            #             print np.all(norm1 == norm2)


                    avgneg = avgneg/length_time
                    avgpos = avgpos/length_time
                    # print(avgneg)
                    # print(avgpos)
                    SMMA = SMMA/length_time
                    try:
                        K = ((aaplc[length_time] - minneg)/(maxposv - minneg))
                    except:
                        K = 6
                    try:
                        RSI = 100 - (100/(1+(avgpos/-avgneg)))
                    except:
                        RSI = 50



                #         print(RSI)
                #     K_.append(K)
                #     RSI_.append(RSI)
                #     OBV_.append(OBV)
                #     N_.append(N)
                #     SMMA_.append(SMMA)


                    return (SMMA, OBV, N, RSI, K, Aroon, MACD, value1, value2, Fish, Value3)

                ##################################################################
                defaultbuy = [30,9,-0.5,-9,-50]
                defaultsell = [70,-20,0.5,-7,50]
                difference  = []
                Weight_List = []
                difference_W  = []
                Weight_List_W = []
                difference_M  = []
                Weight_List_M = []

                def Output(K_,RSI_,OBV_,N_,SMMA_,Aroon_,MACD_,Fish_, i):
                    if i == 0:
                        p = 0
                    if i == 1:
                        p = 1
                    else:
                        p = i-2
                    K = K_[p]
                    RSI = RSI_[p]
                    OBV = OBV_[p]
                    N = N_[p]
                    SMMA = SMMA_[p]
                    Aroon = Aroon_[p]
                    MACD = MACD_[p]
                    Fish = Fish_[p]

                    def Rule_1(RSI):
                        if RSI < defaultbuy[0]:
                            Rule_1_output = 1
                #             print('Rule1')
                        else:
                            return (weight1, 0)

                        if Rule_1_output == True:
                            return (weight1, Rule_1_output)
                        else:
                            return (weight1, Rule_1_output)

                    def Rule_2(stock):
                        if K < defaultbuy[1]:
                            Rule_2_output = 1
                #             print('Rule2')
                        else:
                            return (weight2, 0)

                        if Rule_2_output == True:
                            return (weight2, Rule_2_output)

                    def Rule_3(stock):
                        if N < defaultbuy[2]:
                            Rule_3_output = 1
                #             print('Rule3')
                        else:
                            return (weight3, 0)

                        if Rule_3_output == True:
                            return (weight3, Rule_3_output)

                    def Rule_4(stock):
                        if OBV < defaultbuy[3]:
                            Rule_4_output = 1
                #             print('Rule4')
                        else:
                            return (weight4, 0)

                        if Rule_4_output == True:
                            return (weight4, Rule_4_output)

                    def Rule_5(stock):
                        if Aroon < -50:
                            Rule_5_output = 1
                #             print('Rule4')
                        else:
                            return (weight5, 0)

                        if Rule_5_output == True:
                            return (weight5, Rule_5_output)

                    def Rule_6(stock):
                        if MACD < 0.5:
                            Rule_6_output = 1
                #             print('Rule4')
                        else:
                            return (weight6, 0)

                        if Rule_6_output == True:
                            return (weight6, Rule_6_output)

            #         def Rule_7(stock):
            #             if Fish < -35:
            #                 Rule_7_output = 1
            #     #             print('Rule4')
            #             else:
            #                 return (weight7, 0)

            #             if Rule_7_output == True:
            #                 return (weight7, Rule_7_output)


                    def Rules(weight1, weight2, weight3, weight4, weight6, Rule_1_output, Rule_2_output, Rule_3_output, Rule_4_output, Rule_6_output):
                        if (Rule_1_output*weight1 + Rule_2_output*weight2 + Rule_3_output*weight3 + Rule_4_output*weight4 + Rule_6_output*weight6)/(weight1 + weight2 + weight3 + weight4 + weight6) > Value:
                            return Buy.append(1)
                        else:
                            return Buy.append(0)


                    def Rule_1_S(RSI):

                        if RSI > defaultsell[0]:
                            Rule_1S_output = 1
                #             print('Rule1')
                        else:
                            return (weight1, 0)

                        if Rule_1S_output == True:
                            return (weight1, Rule_1S_output)
                        else:
                            return (weight1, Rule_1S_output)

                    def Rule_2_S(stock):
                        if K > defaultsell[1]:
                            Rule_2S_output = 1
                #             print('Rule2')
                        else:
                            return (weight2, 0)

                        if Rule_2S_output == True:
                            return (weight2, Rule_2S_output)

                    def Rule_3_S(stock):
                        if N > defaultsell[2]:
                            Rule_3S_output = 1
                #             print('Rule3')
                        else:
                            return (weight3, 0)

                        if Rule_3S_output == True:
                            return (weight3, Rule_3S_output)

                    def Rule_4_S(stock):
                        if OBV > defaultsell[3]:
                            Rule_4S_output = 1
                #             print('Rule4')
                        else:
                            return (weight4, 0)

                        if Rule_4S_output == True:
                            return (weight4, Rule_4S_output)

                    def Rule_6_S(stock):
                        if MACD > 0.5:
                            Rule_6S_output = 1
                #             print('Rule4')
                        else:
                            return (weight6, 0)

                        if Rule_6S_output == True:
                            return (weight6, Rule_6S_output)

            #         def Rule_7_S(stock):
            #             if Fish > -10:
            #                 Rule_7S_output = 1
            #     #             print('Rule4')
            #             else:
            #                 return (weight7, 0)

            #             if Rule_7S_output == True:
            #                 return (weight7, Rule_7S_output)


                    def Rules_S(weight1, weight2, weight3, weight4, weight6, Rule_1S_output, Rule_2S_output, Rule_3S_output, Rule_4S_output, Rule_6S_output):
                        if (Rule_1S_output*weight1 + Rule_2S_output*weight2 + Rule_3S_output*weight3 + Rule_4S_output*weight4 + Rule_6S_output*weight6)/(weight1 + weight2 + weight3 + weight4 + weight6) > Value:
                            return Sell.append(1)
                        else:
                            return Sell.append(0)

                    Rule_1_output = Rule_1(RSI)[1]
                    Rule_2_output = Rule_2(K)[1]
                    Rule_3_output = Rule_3(N)[1]
                    Rule_4_output = Rule_4(OBV)[1]
                    Rule_6_output = Rule_6(MACD)[1]
            #         Rule_7_output = Rule_7(Fish)[1]

                    Rules(weight1, weight2, weight3, weight4, weight6, Rule_1_output, Rule_2_output, Rule_3_output, Rule_4_output, Rule_6_output)

                    Rule_1S_output = Rule_1_S(RSI)[1]
                    Rule_2S_output = Rule_2_S(K)[1]
                    Rule_3S_output = Rule_3_S(N)[1]
                    Rule_4S_output = Rule_4_S(OBV)[1]
                    Rule_6S_output = Rule_6_S(MACD)[1]
            #         Rule_7S_output = Rule_7_S(Fish)[1]


                    Rules_S(weight1, weight2, weight3, weight4, weight6, Rule_1S_output, Rule_2S_output, Rule_3S_output, Rule_4S_output, Rule_6S_output)

                    Weight_Value = 3+(((((RSI/100)+ Rule_1_output*weight1) + (((K+20)/20) + Rule_2_output*weight2) + (((N+1)/2)+Rule_3_output*weight3) + Rule_4_output*weight4 + Rule_6_output*weight6)/(weight1 + weight2 + weight3 + weight4 + weight6)) + ((((RSI/100)+Rule_1S_output*weight1) + (((K+20)/20)+Rule_2S_output*weight2) + (((N+1)/2)+Rule_3S_output*weight3) + Rule_4S_output*weight4 + Rule_6S_output*weight6)/(weight1 + weight2 + weight3 + weight4 + weight6)))/2
                    Weight_List.append(Weight_Value)
                    Weight_Value_W = 3+(((((RSI/100)+ Rule_1_output*weight1) + (((K+20)/20) + Rule_2_output*weight2) + (((N+1)/2)+Rule_3_output*weight3) + Rule_4_output*weight4 + Rule_6_output*weight6)/(weight1 + weight2 + weight3 + weight4 + weight6)) + ((((RSI/100)+Rule_1S_output*weight1) + (((K+20)/20)+Rule_2S_output*weight2) + (((N+1)/2)+Rule_3S_output*weight3) + Rule_4S_output*weight4 + Rule_6S_output*weight6)/(weight1 + weight2 + weight3 + weight4 + weight6)))/2
                    Weight_List_W.append(Weight_Value)
                    Weight_Value_M = 3+(((((RSI/100)+ Rule_1_output*weight1) + (((K+20)/20) + Rule_2_output*weight2) + (((N+1)/2)+Rule_3_output*weight3) + Rule_4_output*weight4 + Rule_6_output*weight6)/(weight1 + weight2 + weight3 + weight4 + weight6)) + ((((RSI/100)+Rule_1S_output*weight1) + (((K+20)/20)+Rule_2S_output*weight2) + (((N+1)/2)+Rule_3S_output*weight3) + Rule_4S_output*weight4 + Rule_6S_output*weight6)/(weight1 + weight2 + weight3 + weight4 + weight6)))/2
                    Weight_List_M.append(Weight_Value)

                def diff():
                    try:
                        w1 = 0.998
                        w2 = 1.002
                        diff = 0
                        bulls = 0
                        bears = 0
                        for i in range(len(Weight_List)):
                            if ((Weight_List[len(Weight_List)-1]) * 0.998) <= Weight_List[i] <= ((Weight_List[len(Weight_List)-1]) * 1.002):
                                if i <= len(aaplc)-5:
                                    difference.append(((((aaplc[i+1] - aaplc[i])/(aaplc[i+1]))*100) + (((aaplc[i+2] - aaplc[i+1])/(aaplc[i+2]))*100) + (((aaplc[i+3] - aaplc[i+2])/(aaplc[i+3]))*100))/3)
                        for i in range(len(difference)):
                            diff += difference[i]
                            if difference[i] > 0:
                                bulls += 1
                            else:
                                bears += 1
                        diff = ((diff/(len(difference)-1)))
                        return (diff, bulls, bears)

                    except:
                        w1 = 0.98
                        w2 = 1.01
                        diff = 0
                        bulls = 0
                        bears = 0
                        for i in range(len(Weight_List)):
                            if ((Weight_List[len(Weight_List)-1]) * 0.998) <= Weight_List[i] <= ((Weight_List[len(Weight_List)-1]) * 1.002):
                                if i <= len(aaplc)-5:
                                    difference.append(((((aaplc[i+1] - aaplc[i])/(aaplc[i+1]))*100) + (((aaplc[i+2] - aaplc[i+1])/(aaplc[i+2]))*100) + (((aaplc[i+3] - aaplc[i+2])/(aaplc[i+3]))*100))/3)
                        for i in range(len(difference)):
                            diff += difference[i]
                            if difference[i] > 0:
                                bulls += 1
                            else:
                                bears += 1
                        diff = ((diff/(len(difference)-1)))
                        return (diff, bulls, bears)

                def diff_W():
                    try:
                        w1 = 0.998
                        w2 = 1.002
                        diffW = 0
                        bull = 0
                        bear = 0
                        for i in range(len(Weight_List_W)):
                            if ((Weight_List_W[len(Weight_List_W)-1]) * 0.998) <= Weight_List_W[i] <= ((Weight_List_W[len(Weight_List_W)-1]) * 1.002):
                                if i <= len(aaplc)-24:
                                    difference_W.append(((aaplc[i+7]- aaplc[i])/(aaplc[i+7])*100)+ (((aaplc[i+14] - aaplc[i+7])/(aaplc[i+14]))*100)  + (((aaplc[i+21] - aaplc[i+14])/(aaplc[i+21]))*100)/3)
                        for i in range(len(difference_W)):
                            diffW += difference_W[i]
                            if differenceW[i] > 0:
                                bull += 1
                            else:
                                bear += 1
                        diffW = ((diffW/(len(difference_W)-1)))
                        return (diffW, bull, bear)

                    except:
                        w1 = 0.98
                        w2 = 1.02
                        diffW = 0
                        bull = 0
                        bear = 0
                        for i in range(len(Weight_List_W)):
                            if ((Weight_List_W[len(Weight_List_W)-1]) * 0.998) <= Weight_List_W[i] <= ((Weight_List_W[len(Weight_List_W)-1]) * 1.002):
                                if i <= len(aaplc)-24:
                                    difference_W.append(((aaplc[i+7]- aaplc[i])/(aaplc[i+7])*100)+ (((aaplc[i+14] - aaplc[i+7])/(aaplc[i+14]))*100)  + (((aaplc[i+21] - aaplc[i+14])/(aaplc[i+21]))*100)/3)
                        for i in range(len(difference_W)):
                            diffW += difference_W[i]
                            if difference_W[i] > 0:
                                bull += 1
                            else:
                                bear += 1
                        diffW = ((diffW/(len(difference_W)-1)))
                        return (diffW, bull, bear)

                def diff_M():
                    try:
                        w1 = 0.998
                        w2 = 1.002
                        diffM = 0
                        bullss = 0
                        bearss = 0
                        for i in range(len(Weight_List_M)):
                            if ((Weight_List_M[len(Weight_List_M)-1]) * 0.9) <= Weight_List_M[i] <= ((Weight_List_M[len(Weight_List_M)-1]) * 1.1):
                                if i <= len(aaplc)-62:
                                    difference_M.append(((aaplc[i+30]- aaplc[i])/(aaplc[i+30])*100)+ (((aaplc[i+60] - aaplc[i+30])/(aaplc[i+60]))*100)/2)
                        for i in range(len(difference_M)):
                            diffM += difference_M[i]
                            if difference_M[i] > 0:
                                bullss += 1
                            else:
                                bearss += 1
                            if len(difference_M) == 0 or len(difference_M) ==1:
                                difference_M.append(1)
                                difference_M.append(1)
                        diffM = ((diffM/(len(difference_M)-1)))
                        return (diffM, bullss, bearss)

                    except:
                        w1 = 0.98
                        w2 = 1.02
                        diffM = 0
                        bullss = 0
                        bearss = 0
                        for i in range(len(Weight_List_M)):
                            if ((Weight_List_M[len(Weight_List_M)-1]) * 0.998) <= Weight_List_M[i] <= ((Weight_List_M[len(Weight_List_M)-1]) * 1.002):
                                if i <= len(aaplc)-62:
                                    difference_M.append(((aaplc[i+30]- aaplc[i])/(aaplc[i+30])*100)+ (((aaplc[i+60] - aaplc[i+30])/(aaplc[i+60]))*100)/2)
                        for i in range(len(difference_M)):
                            diffM += difference_M[i]
                            if difference_M[i] > 0:
                                bullss += 1
                            else:
                                bearss += 1
                            if len(difference_M) == 0 or len(difference_M) ==1:
                                difference_M.append(1)
                                difference_M.append(1)
                        diffM = ((diffM/(len(difference_M)-1)))
                        return (diffM, bullss, bearss)
                ##################################################################

                indic = PSAR()
                data['PSAR'] = data.apply(lambda x: indic.calcPSAR(x['High'], x['Low']), axis=1)
                data['EP'] = indic.ep_list
                data['Trend'] = indic.trend_list
                data['AF'] = indic.af_list
                data.head()
                indic._calcPSAR()

                value2 = 14
                value4 = value2
            #     for i in range(4):
            #         stock.append(stock[len(stock)-1])

                for i in range(16):
            #         stock.append(stock[len(stock)-1])
                    aaplc.append(aaplc[len(aaplc)-1])
                    aaplo.append(aaplo[len(aaplo)-1])
                    aaplh.append(aaplh[len(aaplh)-1])
                    aapll.append(aapll[len(aapll)-1])
                    aaplv.append(aaplv[len(aaplv)-1])

            #         print(len(aaplc))

                for i in range(len(stock)-value2):
                    value1 = value2 + i

            #             print(Indicator(value1, value2))
                    if i >= len(stock)-value2-14:
                        burned_variable = []
                        burned_variable_W = []
                        burned_variable.append(diff())
                        burned_variable_W.append(diff_W())
            #             print(aaplc[i-1] * ((burned_variable[0][0] + burned_variable_W[0][0]/7)/2)/100)
            #             print(aaplc[i])
        #                 aaplc[i] = aaplc[i-1]
        #                 aaplo[i] = aaplo[i-1]
        #                 aaplh[i] = aaplh[i-1]
        #                 aapll[i] = aapll[i-1]
        #                 aaplv[i] = aaplv[i-1]
            #             print(aaplc[i])
            #             print(aaplc[len(aaplc)-1])

                    RSI = Indicator(value1, value2, Fish, Value3)[3]
                    K = Indicator(value1, value2,Fish, Value3)[4]
                    OBV = Indicator(value1, value2,Fish, Value3)[1]
                    N = Indicator(value1, value2,Fish, Value3)[2]
                    SMMA = Indicator(value1, value2,Fish, Value3)[0]
                    Aroon = Indicator(value1, value2,Fish, Value3)[5]
                    MACD = Indicator(value1, value2,Fish, Value3)[6]
                    Williams = Indicator(value1, value2,Fish, Value3)[7]
                    Fish = Indicator(value1, value2,Fish, Value3)[9]
                    Value3 = Indicator(value1, value2,Fish, Value3)[10]

            #         print(value1)
            #         print(i)
            #         print(len(stock))

                    K_.append(K)
                    RSI_.append(RSI)
                    OBV_.append(OBV)
                    N_.append(N)
                    SMMA_.append(SMMA)
                    Aroon_.append(Aroon)
                    MACD_.append(MACD)
                    Fish_.append(Fish)
                    Value3_.append(Value3)
                    FishI.append(Fish)


                    if i == len(stock)-value2-1:
        #                 for p in range(8):
        #                     K_.insert(0,0)
        #                     RSI_.insert(0,0)
        #                     OBV_.insert(0,0)
        #                     N_.insert(0,0)
        #                     SMMA_.insert(0,0)
        #                     Aroon_.insert(0,0)
        #                     MACD_.insert(0,0)
        #                     Fish_.insert(0,0)
        #                     Value3_.insert(0,0)
                        for p in range(2):
                            K_.pop(len(K_)-1)
                            RSI_.pop(len(RSI_)-1)
                            OBV_.pop(len(OBV_)-1)
                            N_.pop(len(N_)-1)
                            SMMA_.pop(len(SMMA_)-1)
                            Aroon_.pop(len(Aroon_)-1)
                            MACD_.pop(len(MACD_)-1)
                            Fish_.pop(len(Fish_)-1)
                            Value3_.pop(len(Value3_)-1)
                            FishI.pop(len(FishI)-1)

            #         psarOutput = 0
            #         if i == (len(stock)-value2):
            #             psar_close = data.loc[data['Trend']==1]['Close']
            #             if psar_close[len(psar_close)-1] == aaplc[len(aaplc)-2]:
            #                 psarOutput = 1

                for i in range(len(stock)-value2):
                    Output(K_,RSI_,OBV_,N_,SMMA_,Aroon_,MACD_,Fish_, i)


                    if Buy[i] == 1:
                        try:
                            K__.append(K_[i])
                            OBV__.append(OBV_[i])
                            RSI__.append(RSI_[i])
                            N__.append(N_[i])
                            MACD__.append(MACD_[i])
            #                 Fish__.append(Fish_[i])
                        except:
                            pass

                Fish_ = [x *-1 for x in Fish_]
                Value3_ = [x *-1 for x in Value3_]
                FishI = [x *-1 for x in FishI]
                diff_ = []
                diff_W_ = []
                diff_M_ = []
                diff_.append(diff())
                diff_W_.append(diff_W())
                diff_M_.append(diff_M())
                bulls = diff_[0][1]
                bears = diff_[0][2]
                bull = diff_W_[0][1]
                bear = diff_W_[0][2]
                bullss = diff_M_[0][1]
                bearss = diff_M_[0][2]
                # Buy = [0,1,1,1,1,1,1]
            #     print(len(stock))
            #     print(len(Stock_Change))
            #     print(len(Buy))
            #     print()
                Fish_.insert(0, 0.5)
                Fish_.insert(0, 0.5)
                Value3_.insert(0, 0.5)
                Value3_.insert(0, 0.5)
                Fish6_ = pd.DataFrame(list(zip(Fish_, Value3_)), columns=['Fish_','Value3_'])

                # Fish6_ = pd.DataFrame (Fish6_, columns = ['Fish_','Value3_'])
                indic = PSAR()

                Fish6_['PSAR'] = Fish6_.apply(lambda x: indic.calcPSAR(x['Fish_'], x['Value3_']), axis=1)
                Fish6_['EP'] = indic.ep_list
                Fish6_['Trend'] = indic.trend_list
                Fish6_['AF'] = indic.af_list
                Fish6_.head()
                indic._calcPSAR()

                fish_psar_bull = Fish6_.loc[Fish6_['Trend']==1]['PSAR']
                fish_psar_bear = Fish6_.loc[Fish6_['Trend']==0]['PSAR']

                for i in range(len(Buy)-1):
                    fish_psar_close = Fish6_['Trend'].iloc[i+1]
                    if Fish_[i] > 1.5:
                        if fish_psar_close == 0 or Fish_[i+1] < Fish_[i]:
                            Sell[i] = 1
                            Sell[i-1] = 1
                            Sell[i-2] = 1

                    if Fish_[i] <= 0:
                        Buy[i] = 1
                        Buy[i-1] = 1
                        Buy[i-2] = 1
                    if i == len(Buy)-2:
                        print(Fish_[i+1])
                        print(Fish_[i])






                def How_Much(Buy, Cash, Stock):
                    value0 = 0
                    value1 = 0.15
                    value2 = 0.35
                    value3 = 0.75
                    j = 0
                    k = 0
                    Stock_Change = []
                    Track_Change = []

                    How_Much_Buy = [0]
                    How_Much_Sell = [0]
                    Cash = 1000
                    Stock = 1000
            #         print(len(Buy))
            #         print(len(stock))

                    for i in range(len(Buy)):
            #                 print(len(Buy))
                    #         print(Stock)

                    #         print(Cash)
                    #         print(Stock)
                    #         print()
            #             f = int(i+value4)

                        f = int(i)
                        if i == 0:
                            stock[f-1] = stock[f]

                        Cash_.append(Cash)
            #             Stock_.append(Stock)

                        Change1 = -(stock[f-1]-stock[f])/(stock[f-1])

                        Change = Stock *(-(stock[f-1]-stock[f])/(stock[f-1]))
                        Stock += Stock * (-(stock[f-1]-stock[f])/(stock[f-1]))
                    #         print(stock[i-1])
                    #         print(stock[i])
                    #         print()
                    #         print(Change1)
                    #         print(Change)
                    #         print()
                        Track_Change.append(Change1)
                        if Stock < 0:
                            How_Much_Buy.append(0)
                            How_Much_Sell.append(0)
                            print('f')
                            continue
                        if Cash < 0:
                            How_Much_Buy.append(0)
                            How_Much_Sell.append(0)
                            print('g')
                            continue

                        if Buy[i] == 1:

                            del How_Much_Sell[k]
                            How_Much_Sell.append(0)
                            if How_Much_Buy[j] == 3:
                                Stock_Change.append('0')
                                pass
                            else:
                                if How_Much_Buy[j] == 0:
                                    if Cash != 0 and Cash > 0:
                                        Stock += Cash*value1
                                        Cash -= Cash*value1
                                        Stock_Change.append('25')

                                if How_Much_Buy[j] == 1:
                                    if Cash != 0 and Cash > 0:
                                        Stock += Cash*value1
                                        Cash -= Cash*value1
                                        Stock_Change.append('55')

                                if How_Much_Buy[j] == 2:
                                    if Cash != 0 and Cash > 0:
                                        Stock += Cash*value2
                                        Cash -= Cash*value2
                                        Stock_Change.append('95')

                                How_Much_Buy.append((How_Much_Buy[j] + 1))
                                j+=1
                                stock[f]

                        if Sell[i] == 1:
                            del How_Much_Buy[j]
                            How_Much_Buy.append(0)
                            if How_Much_Sell[k] == 3:
                                Stock_Change.append('0')
                                pass
                            else:
                                if How_Much_Sell[k] == 0:
                                    if Stock != 0 and Stock > 0:
                                        Cash += Stock*value1
                                        Stock -= Stock*value1
                                        Stock_Change.append('-25')

                                if How_Much_Sell[k] == 1:
                                    if Stock != 0 and Stock > 0:
                                        Cash += Stock*value2
                                        Stock -= Stock*value2
                                        Stock_Change.append('-55')

                                if How_Much_Sell[k] == 2:
                                    if Stock != 0 and Stock > 0:
                                        Cash += Stock*value3
                                        Stock -= Stock*value3
                                        Stock_Change.append('-95')
                    #                 if Stock < 0:
                    #                     return

                                How_Much_Sell.append((How_Much_Sell[k] + 1))
                                k+=1
                        if Buy[i] == 0 and Sell[i] == 0:
                            Stock_Change.append('0')
                        day.append(i)
                        account_dict["Stock"].append(Stock)
                        account_dict["Cash"].append(Cash)
                    Total = Stock + Cash
                    return (Total, Stock_Change, Stock, Cash, Track_Change)

            #     print(len(Stock_Change))
            #     print(len(day))
            #     print(len(Buy))
                List = []
                List.append(How_Much(Buy, Cash, Stock))
                Stock_Change = List[0][1]
                Stock_Exchange.append(Stock_Change)
                Stock = List[0][2]
                Cash = List[0][3]
                Stocks_Change.append(List[0][4])
                Stock_Names.append(Stock_Name)

                CS = Cash/Stock

                # if o == 0:
                #     aaplvv = aaplv[len(aaplv)-2]
                #     aaploo = aaplo[len(aaplo)-2]
                #     aaplcc = aaplc[len(aaplc)-2]
                #     aaplll = aapll[len(aapll)-2]
                #     aaplhh = aaplh[len(aaplh)-2]

                ################################################################
            #     MUTLISTOCK ARBITRAGE
                ################################################################

            #     Stock1 = 0
            #     Cash1 = 1000
            #     for i in range(length):
            #         Stock2 = 0
            #         for j in range(len(Stock_Exchange)):
            #             if int(Stock_Exchange[j][i]) != 0 and int(Stock_Exchange[j][i]) > 0:
            #                 if Cash1 != 0 and Cash1 > 0:
            #                     Stock_[j][i] += Cash1*(int(Stock_Exchange[j][i])/100)
            #                     Cash1 -= Cash1*(int(Stock_Exchange[j][i])/100)
            #                     Stock_[j][i] += Stock_[j][i] * float(Stocks_Change[j][i+1])
            #                     try:
            #                         Stock_[j][i+1] = (Stock_[j][i])
            #                     except:
            #                         pass
            #             if int(Stock_Exchange[j][i]) != 0 and int(Stock_Exchange[j][i]) < 0:
            #                 if Stock_[j][i] != 0 and Stock_[j][i] > 0:
            #                     Cash1 += (Stock_[j][i]*(-int(Stock_Exchange[j][i])/100))
            #                     Stock_[j][i] -= (Stock_[j][i]*(-int(Stock_Exchange[j][i])/100))
            #                     Stock_[j][i] += (Stock_[j][i]*float(Stocks_Change[j][i+1]))
            #                     try:
            #                         Stock_[j][i+1] = (Stock_[j][i])
            #                     except:
            #                         pass
            #             if int(Stock_Exchange[j][i]) == 0:
            #                     Stock_[j][i] += Stock_[j][i]*float(Stocks_Change[j][i])
            #                     try:
            #                         Stock_[j][i+1] = (Stock_[j][i])
            #                     except:
            #                         pass
            #             Stock2 += Stock_[j][i]
            #             if j == len(Stock_Exchange)-1:
            #                 Stock1 = Stock2

            #         account_dict1["Stock1"].append(Stock1)
            #         account_dict1["Cash1"].append(Cash1)
            #     #E36D70
            # #     color_map = ["#E36D70", "#0DC664"]
            #     color_map = ["#0FF415", "#2268A7"]#2268A7
            #     fig, ax = plt.subplots(figsize=(12,3))
            #     ax.stackplot(day, account_dict1.values(), labels=account_dict.keys(),colors = color_map)
            #     ax.legend(loc='upper left')
            #     plt.yscale("log")
            #     plt.show()
             ##################################################################



            ################################################################
                print()
                print(Stock_Name)
                print("Cash/Stock Ratio " + str(CS)[:-13])
                print("You should Buy/Sell this percentage last 10 days " +   Stock_Change[len(Stock_Change)- 18]+ ' ' + Stock_Change[len(Stock_Change)- 17] + ' ' + Stock_Change[len(Stock_Change)- 16] + ' ' + Stock_Change[len(Stock_Change)- 15]  + ' ' + Stock_Change[len(Stock_Change)- 14] + ' ' + Stock_Change[len(Stock_Change)- 13] + ' ' + Stock_Change[len(Stock_Change)- 12]+ ' ' + Stock_Change[len(Stock_Change)- 11] + ' ' + Stock_Change[len(Stock_Change)- 10] + ' ' + Stock_Change[len(Stock_Change)- 9] + ' ' + Stock_Change[len(Stock_Change)- 8] + ' ' + Stock_Change[len(Stock_Change)- 7] + ' ' + Stock_Change[len(Stock_Change)- 6] + ' ' + Stock_Change[len(Stock_Change)- 5] +  ' ' + Stock_Change[len(Stock_Change)- 4] +  ' ' + Stock_Change[len(Stock_Change)- 3] + ' ' + Stock_Change[len(Stock_Change)- 2] + ' ' + Stock_Change[len(Stock_Change)- 1])
            #     print("The Daily Expected Move for Tommorrow is " + str(diff_[0][0])[:-14] + ' bulls: ' + str(bulls) + ' bears: ' + str(bears))
            #     print("The Weekly Expected Move for This Week is " + str(diff_W_[0][0])[:-14] + ' bulls: ' + str(bull) + ' bears: ' + str(bear))
            #     print("The Monthly Expected Move for This Month is " + str(diff_M_[0][0])[:-14] + ' bulls: ' + str(bullss) + ' bears: ' + str(bearss))

                #     color_map = ["#BBA155", "#0C4808"]
            #     fig, ax = plt.subplots(figsize=(12,2))
            #     ax.stackplot(day, account_dict.values(), labels=account_dict.keys(),colors = color_map)
            #     ax.legend(loc='upper left')
            #     plt.yscale("linear")
            #     plt.show()
            ##########################################################
                psar_close = data.loc[data['Trend']==1]['Close']


                if CS < 2:
                    if psar_close[len(psar_close)-1] == aaplc[len(aaplc)-2]:
                        if int(Stock_Change[len(Stock_Change)- 10]) + int(Stock_Change[len(Stock_Change)- 9]) + int(Stock_Change[len(Stock_Change)- 8]) + int(Stock_Change[len(Stock_Change)- 7]) + int(Stock_Change[len(Stock_Change)- 6]) + int(Stock_Change[len(Stock_Change)- 5]) + int(Stock_Change[len(Stock_Change)- 4]) + int(Stock_Change[len(Stock_Change)- 3]) + int(Stock_Change[len(Stock_Change)- 2]) + int(Stock_Change[len(Stock_Change)- 1]) + int(Stock_Change[len(Stock_Change)-11]) > 0:
                            CS_.append(Stock_Name)
                            CS_.append(CS)
                            CS_.append(Stock+Cash)
                if diff_W_[0][0] > 10:
                    if psar_close[len(psar_close)-1] == aaplc[len(aaplc)-2]:
                        if int(Stock_Change[len(Stock_Change)- 10]) + int(Stock_Change[len(Stock_Change)- 9]) + int(Stock_Change[len(Stock_Change)- 8]) + int(Stock_Change[len(Stock_Change)- 7]) + int(Stock_Change[len(Stock_Change)- 6]) + int(Stock_Change[len(Stock_Change)- 5]) + int(Stock_Change[len(Stock_Change)- 4]) + int(Stock_Change[len(Stock_Change)- 3]) + int(Stock_Change[len(Stock_Change)- 2]) + int(Stock_Change[len(Stock_Change)- 1]) + int(Stock_Change[len(Stock_Change)- 11]) > 0:
                            CS_.append(Stock_Name)
                            CS_.append(CS)
                            CS_.append(Stock+Cash)

                # if Value3_[len(Value3_)-1] < Value3_[len(Value3_)-2]:
                #     if Value3_ < Fish_ or Fish_[len(Fish_)-1] < Fish_[len(Fish_)-2]:
                #         FishIndicator.append(1)

                Monthly.append(float(diff_M_[0][0]))
                Weekly.append(float(diff_W_[0][0]))
        #
                # if CS < 2 or CS > 2:
        #     #     if int(Stock_Change[len(Stock_Change)-1]) != 0 or int(Stock_Change[len(Stock_Change)-2]) != 0 or int(Stock_Change[len(Stock_Change)-3]) != 0:
        #     #     if True:
        #             print(Stock_Name)
        #             print("Stock Close " + str(aaplc[len(aaplc)-2]))
        #             print("Stock Volatility " + str(volatility*100))
        #             print("Stock " + str(Stock)[:-9])
        #             print("Cash " + str(Cash)[:-9])
        #             print("Cash/Stock Ratio " + str(CS)[:-13])
        #     #         print("The Daily Expected Move for Tommorrow is " + str(diff_[0][0])[:-10] + ' bulls: ' + str(bulls) + ' bears: ' + str(bears))
        #     #         print("The Weekly Expected Move for This Week is " + str(diff_W_[0][0])[:-10]  + ' bulls: ' + str(bull) + ' bears: ' + str(bear))
        #             print("You should Buy/Sell this percentage last 10 days " + Stock_Change[len(Stock_Change)- 10] + ' ' + Stock_Change[len(Stock_Change)- 9] + ' ' + Stock_Change[len(Stock_Change)- 8] + ' ' + Stock_Change[len(Stock_Change)- 7] + ' ' + Stock_Change[len(Stock_Change)- 6] + ' ' + Stock_Change[len(Stock_Change)- 5] +  ' ' + Stock_Change[len(Stock_Change)- 4] +  ' ' + Stock_Change[len(Stock_Change)- 3] + ' ' + Stock_Change[len(Stock_Change)- 2] + ' ' + Stock_Change[len(Stock_Change)- 1])
        #
        #             import matplotlib.pyplot as plt
        #
        #             colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        #
        #             psar_bull = data.loc[data['Trend']==1]['PSAR']
        #             psar_bear = data.loc[data['Trend']==0]['PSAR']
        #             plt.figure(figsize= (16, 8))
        #             plt.plot(data['Close'], label='Close', linewidth=1)
        # #             result = [i1 * i2 for i1, i2 in zip(list_1, list_2)]
        #
        #             plt.scatter(psar_bull.index, psar_bull, color=colors[1], label='Up Trend')
        #             plt.scatter(psar_bear.index, psar_bear, color=colors[3], label='Down Trend')
        #             plt.xlabel('Date')
        #             plt.ylabel('Price ($)')
        #             plt.title(f'{Stock_Name}')
        #             plt.legend()
        #
        #             plt.show()
        #
        #
        #             plt.figure(figsize= (16, 8))
        #             # plt.plot(Fish_)
        #             plt.scatter(fish_psar_bull.index, fish_psar_bull, color=colors[1], label='Up Trend')
        #             plt.scatter(fish_psar_bear.index, fish_psar_bear, color=colors[3], label='Down Trend')
        #
        #             plt.plot(Fish_)
        #             plt.plot(Value3_)
        #             plt.show
        #
        #             plt.figure(figsize= (16, 4))
        #
        #             width = 0.1
        #             width2 = 0.1
        #
        #             up = aapl[aapl.Close>=aapl.Open]
        #             down = aapl[aapl.Close<aapl.Open]
        #
        #             col1 = 'green'
        #             col2 = 'red'
        #
        #             plt.bar(up.index,up.Close-up.Open,width,bottom=up.Open,color=col1)
        #             plt.bar(up.index,up.High-up.Close,width2,bottom=up.Close,color=col1)
        #             plt.bar(up.index,up.Low-up.Open,width2,bottom=up.Open,color=col1)
        #
        #             #plot down prices
        #             plt.bar(down.index,down.Close-down.Open,width,bottom=down.Open,color=col2)
        #             plt.bar(down.index,down.High-down.Open,width2,bottom=down.Open,color=col2)
        #             plt.bar(down.index,down.Low-down.Close,width2,bottom=down.Close,color=col2)
        #
        #             #rotate x-axis tick labels
        #             plt.xticks(rotation=45, ha='right')
        #
        #             #display candlestick chart
        #             plt.show()
        #
        #             color_map = ["#BBA155", "#0C4808"]
        # #
        #             fig, ax = plt.subplots(figsize=(16,4))
        #             ax.stackplot(day, account_dict.values(), labels=account_dict.keys(),colors = color_map)
        #             ax.legend(loc='upper left')
        #             plt.yscale("linear")
        #
        #             plt.show()
                # if fish_psar_close == 0:
                #     FishIndicator -=0
                #
                # if Value3_[len(Value3_)-1] < Value3_[len(Value3_)-2]:
                #     if Value3_ < Fish_ or Fish_[len(Fish_)-1] < Fish_[len(Fish_)-2]:
                #         FishIndicator += 1*o
                #
                # if Value3_[len(Value3_)-1] > Value3_[len(Value3_)-2]:
                #     if Value3_ > Fish_ or Fish_[len(Fish_)-1] > Fish_[len(Fish_)-2]:
                #         FishIndicator -= 1*o

##############################################################
                common_length = min(len(K_), len(RSI_), len(OBV_), len(N_), len(SMMA_), len(Aroon_), len(MACD_), len(Fish_), len(Value3_), len(aaplv), len(aaplo), len(aaplc), len(aapll), len(aaplh))

                # Resize each array to common_length by excluding the beginning indices
                K_array = np.array(K_[-common_length:])
                RSI_array = np.array(RSI_[-common_length:])
                OBV_array = np.array(OBV_[-common_length:])
                N_array = np.array(N_[-common_length:])
                SMMA_array = np.array(SMMA_[-common_length:])
                Aroon_array = np.array(Aroon_[-common_length:])
                MACD_array = np.array(MACD_[-common_length:])
                Fish_array = np.array(Fish_[-common_length:])
                Value3_array = np.array(Value3_[-common_length:])
                aaplv_array = np.array(aaplv[-common_length:])
                aaplo_array = np.array(aaplo[-common_length:])
                aaplc_array = np.array(aaplc[-common_length:])
                aapll_array = np.array(aapll[-common_length:])
                aaplh_array = np.array(aaplh[-common_length:])
                Buy_array = np.array(Buy[-common_length:])


                # Stack the individual feature arrays horizontally to create the feature array
                feature_arrays = np.column_stack((K_array, RSI_array, OBV_array, N_array, SMMA_array, Aroon_array,
                                                  MACD_array, Fish_array, Value3_array,
                                                  aaplv_array, aaplo_array, aaplc_array, aapll_array, aaplh_array))
                # lables = np.column_stack(Buy)

                # Generate dummy labels for testing

                # Now, you can use these arrays as features in your machine learning model
                y = np.array(Buy_array)

                # Split the data into training and testing sets
                X_train, X_test, y_train, y_test = train_test_split(feature_arrays, y, test_size=0.2, random_state=42)

                # Standardize the features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                # Calculate Lorentzian Distance matrix between training and testing sets
                distance_matrix_train = cdist(X_train_scaled, X_train_scaled, metric='euclidean')
                X_train_ld = np.sum(np.log(1 + np.abs(distance_matrix_train)), axis=1)

                distance_matrix_test = cdist(X_test_scaled, X_train_scaled, metric='euclidean')
                X_test_ld = np.sum(np.log(1 + np.abs(distance_matrix_test)), axis=1)

                # Create and train the KNN model
                knn_model = KNeighborsClassifier(n_neighbors=5, metric='precomputed')  # Use 'precomputed' for custom distance metrics
                knn_model.fit(distance_matrix_train, y_train)

                # Make predictions on the test set
                y_pred = knn_model.predict(distance_matrix_test)

                # Evaluate the accuracy
                accuracy = accuracy_score(y_test, y_pred)
                print(f'Accuracy: {accuracy}')
                import matplotlib.pyplot as plt

                # Plot actual vs. predicted outcomes
                plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual')
                plt.scatter(range(len(y_test)), y_pred, color='red', label='Predicted')
                plt.xlabel('Sample Index')
                plt.ylabel('Outcome')
                plt.title('Actual vs. Predicted Outcomes')
                plt.legend()
                plt.show()

                # Get the last values for each feature
                K_current = K_array[-1]
                RSI_current = RSI_array[-1]
                OBV_current = OBV_array[-1]
                N_current = N_array[-1]
                SMMA_current = SMMA_array[-1]
                Aroon_current = Aroon_array[-1]
                MACD_current = MACD_array[-1]
                Fish_current = Fish_array[-1]
                Value3_current = Value3_array[-1]
                aaplv_current = aaplv_array[-1]
                aaplo_current = aaplo_array[-1]
                aaplc_current = aaplc_array[-1]
                aapll_current = aapll_array[-1]
                aaplh_current = aaplh_array[-1]

                # Print the current day features
                print("Current Day Features:")
                print(f"K: {K_current}")
                print(f"RSI: {RSI_current}")
                print(f"OBV: {OBV_current}")
                print(f"N: {N_current}")
                print(f"SMMA: {SMMA_current}")
                print(f"Aroon: {Aroon_current}")
                print(f"MACD: {MACD_current}")
                print(f"Fish: {Fish_current}")
                print(f"Value3: {Value3_current}")
                print(f"aaplv: {aaplv_current}")
                print(f"aaplo: {aaplo_current}")
                print(f"aaplc: {aaplc_current}")
                print(f"aapll: {aapll_current}")
                print(f"aaplh: {aaplh_current}")


                current_day_features = np.array([K_current, RSI_current, OBV_current, N_current, SMMA_current, Aroon_current,
                                          MACD_current, Fish_current, Value3_current,
                                          aaplv_current, aaplo_current, aaplc_current, aapll_current, aaplh_current])

                # Standardize the features
                current_day_features_scaled = scaler.transform(current_day_features.reshape(1, -1))

                # Calculate Lorentzian Distance matrix between the current day and training set
                distance_matrix_current_day = cdist(current_day_features_scaled, X_train_scaled, metric='euclidean')
                current_day_ld = np.sum(np.log(1 + np.abs(distance_matrix_current_day)), axis=1)

                # Use the trained KNN model to predict the outcome for the current day
                predicted_outcome = knn_model.predict(distance_matrix_current_day)

                print(f'Predicted Buy Outcome for the Current Day: {predicted_outcome[0]}')





                ####################################################################################

                common_length = min(len(K_), len(RSI_), len(OBV_), len(N_), len(SMMA_), len(Aroon_), len(MACD_), len(Fish_), len(Value3_), len(aaplv), len(aaplo), len(aaplc), len(aapll), len(aaplh))

                # Resize each array to common_length by excluding the beginning indices
                K_array = np.array(K_[-common_length:])
                RSI_array = np.array(RSI_[-common_length:])
                OBV_array = np.array(OBV_[-common_length:])
                N_array = np.array(N_[-common_length:])
                SMMA_array = np.array(SMMA_[-common_length:])
                Aroon_array = np.array(Aroon_[-common_length:])
                MACD_array = np.array(MACD_[-common_length:])
                Fish_array = np.array(Fish_[-common_length:])
                Value3_array = np.array(Value3_[-common_length:])
                aaplv_array = np.array(aaplv[-common_length:])
                aaplo_array = np.array(aaplo[-common_length:])
                aaplc_array = np.array(aaplc[-common_length:])
                aapll_array = np.array(aapll[-common_length:])
                aaplh_array = np.array(aaplh[-common_length:])
                Sell_array = np.array(Sell[-common_length:])


                # Stack the individual feature arrays horizontally to create the feature array
                feature_arrays = np.column_stack((K_array, RSI_array, OBV_array, N_array, SMMA_array, Aroon_array,
                                                  MACD_array,Fish_array, Value3_array,
                                                  aaplv_array, aaplo_array, aaplc_array, aapll_array, aaplh_array))
                y = np.array(Sell_array)

                # Split the data into training and testing sets
                X_train, X_test, y_train, y_test = train_test_split(feature_arrays, y, test_size=0.2, random_state=42)

                # Standardize the features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                # Calculate Lorentzian Distance matrix between training and testing sets
                distance_matrix_train = cdist(X_train_scaled, X_train_scaled, metric='euclidean')
                X_train_ld = np.sum(np.log(1 + np.abs(distance_matrix_train)), axis=1)

                distance_matrix_test = cdist(X_test_scaled, X_train_scaled, metric='euclidean')
                X_test_ld = np.sum(np.log(1 + np.abs(distance_matrix_test)), axis=1)

                # Create and train the KNN model
                knn_model = KNeighborsClassifier(n_neighbors=5, metric='precomputed')  # Use 'precomputed' for custom distance metrics
                knn_model.fit(distance_matrix_train, y_train)

                # Make predictions on the test set
                y_pred = knn_model.predict(distance_matrix_test)

                # Evaluate the accuracy
                accuracy = accuracy_score(y_test, y_pred)
                print(f'Accuracy: {accuracy}')
                import matplotlib.pyplot as plt

                # Plot actual vs. predicted outcomes
                plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual')
                plt.scatter(range(len(y_test)), y_pred, color='red', label='Predicted')
                plt.xlabel('Sample Index')
                plt.ylabel('Outcome')
                plt.title('Actual vs. Predicted Outcomes')
                plt.legend()
                plt.show()

                # Get the last values for each feature
                K_current = K_array[-1]
                RSI_current = RSI_array[-1]
                OBV_current = OBV_array[-1]
                N_current = N_array[-1]
                SMMA_current = SMMA_array[-1]
                Aroon_current = Aroon_array[-1]
                MACD_current = MACD_array[-1]
                Fish_current = Fish_array[-1]
                Value3_current = Value3_array[-1]
                aaplv_current = aaplv_array[-1]
                aaplo_current = aaplo_array[-1]
                aaplc_current = aaplc_array[-1]
                aapll_current = aapll_array[-1]
                aaplh_current = aaplh_array[-1]

                # Print the current day features
                print("Current Day Features:")
                print(f"K: {K_current}")
                print(f"RSI: {RSI_current}")
                print(f"OBV: {OBV_current}")
                print(f"N: {N_current}")
                print(f"SMMA: {SMMA_current}")
                print(f"Aroon: {Aroon_current}")
                print(f"MACD: {MACD_current}")
                print(f"aaplv: {aaplv_current}")
                print(f"aaplo: {aaplo_current}")
                print(f"aaplc: {aaplc_current}")
                print(f"aapll: {aapll_current}")
                print(f"aaplh: {aaplh_current}")
                print("length of features")
                print("length of features" f'{len(K_array)}')
                print("length of Buy array" f'{len(Buy_array)}')


                current_day_features = np.array([K_current, RSI_current, OBV_current, N_current, SMMA_current, Aroon_current,
                                          MACD_current, Fish_current, Value3_current,
                                          aaplv_current, aaplo_current, aaplc_current, aapll_current, aaplh_current])

                # Standardize the features
                current_day_features_scaled = scaler.transform(current_day_features.reshape(1, -1))

                # Calculate Lorentzian Distance matrix between the current day and training set
                distance_matrix_current_day = cdist(current_day_features_scaled, X_train_scaled, metric='euclidean')
                current_day_ld = np.sum(np.log(1 + np.abs(distance_matrix_current_day)), axis=1)

                # Use the trained KNN model to predict the outcome for the current day
                predicted_outcome_sell = knn_model.predict(distance_matrix_current_day)

                print(f'Predicted Sell Outcome for the Current Day: {predicted_outcome_sell[0]}')

                import matplotlib.pyplot as plt

                colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

                psar_bull = data.loc[data['Trend']==1]['PSAR']
                psar_bear = data.loc[data['Trend']==0]['PSAR']
                plt.figure(figsize= (16, 8))
                plt.plot(data['Close'], label='Close', linewidth=1)
                #             result = [i1 * i2 for i1, i2 in zip(list_1, list_2)]
                # indices = np.arange((y_test))
                # indices_sell = np.arange((y_test))

                print("Right Here Bitch")
                print(y_pred)
                print(len(y_pred))
                print("Right Here Bitch")

                print(psar_bull.index)
                print(len(psar_bull.index))
                print("Right Here Bitch")


                plt.scatter(psar_bull.index, psar_bull, color=colors[1], label='Up Trend')
                plt.scatter(psar_bear.index, psar_bear, color=colors[3], label='Down Trend')

                plt.scatter(y_pred.index, y_pred, color=colors[1])
                plt.scatter(y_pred.index, y_pred, color=colors[3])

                plt.xlabel('Date')
                plt.ylabel('Price ($)')
                plt.title(f'{Stock_Name}')
                plt.legend()

                plt.show()


                # plt.figure(figsize= (16, 8))
                # # plt.plot(Fish_)
                # plt.scatter(fish_psar_bull.index, fish_psar_bull, color=colors[1], label='Up Trend')
                # plt.scatter(fish_psar_bear.index, fish_psar_bear, color=colors[3], label='Down Trend')
                #
                # plt.plot(Fish_)
                # plt.plot(Value3_)
                # plt.show()

                plt.figure(figsize= (16, 4))

                width = 0.25
                width2 = 0.25

                up = aapl[aapl.Close>=aapl.Open]
                down = aapl[aapl.Close<aapl.Open]

                col1 = 'green'
                col2 = 'red'

                plt.bar(up.index,up.Close-up.Open,width,bottom=up.Open,color=col1)
                plt.bar(up.index,up.High-up.Close,width2,bottom=up.Close,color=col1)
                plt.bar(up.index,up.Low-up.Open,width2,bottom=up.Open,color=col1)

                #plot down prices
                plt.bar(down.index,down.Close-down.Open,width,bottom=down.Open,color=col2)
                plt.bar(down.index,down.High-down.Open,width2,bottom=down.Open,color=col2)
                plt.bar(down.index,down.Low-down.Close,width2,bottom=down.Close,color=col2)

                #rotate x-axis tick labels
                plt.xticks(rotation=45, ha='right')


                print()

                #display candlestick chart
                plt.show()
                #
                color_map = ["#BBA155", "#0C4808"]
                #
                fig, ax = plt.subplots(figsize=(16,4))
                ax.stackplot(day, account_dict.values(), labels=account_dict.keys(),colors = color_map)
                ax.legend(loc='upper left')
                plt.yscale("linear")

                plt.show()




            # ###################################
                if b == 0:
                    if o == 1:
                        try:
                            btc = Stock_Name
                            Stock_Name = [btc]
                            CSA = 0
                            account = alpaca.get_account()
                            account_cash = float(account.cash)
                            account_equity = float(account.equity)
                            try:
                                position = alpaca.get_position(btc).qty
                            except:
                                position = 0

                            cash_to_spend = (account_cash-5000)/ 1.1

                            # btc_snapshot = alpaca.get_crypto_snapshot(symbol=btc, exchange="CBSE")
                            btc_snapshot = alpaca.get_snapshots(Stock_Name)
                            btc_latest_price = aaplc[len(aaplc)-2]
                            print('latest price:')
                            print(btc_latest_price)
                            print()
                            # btc_latest_price = btc_snapshot.latest_quote.ap
                            bs = ""
                            def calculate_order_size(cash_to_spend, latest_price, position, CST, Cash, Stock, volatility):
                            #     change = 0
                            #     psar_boost = 0.8
                            #     if psar_close[len(psar_close)-1] == aaplc[len(aaplc)-2]:
                            #         psar_boost = 1.2
                            #     print(CS)
                            #
                            #     if CS < 1:
                            #         change = 0.35 * psar_boost
                            #         bs = "buy"
                            #     if CS > 1:
                            #         change = 0.85
                            #         bs = "sell"
                            #
                            #     if ( int(Stock_Change[len(Stock_Change)- 3]) + int(Stock_Change[len(Stock_Change)- 2]) + int(Stock_Change[len(Stock_Change)- 1])) > 0:
                            #         if change == 0:
                            #             if change < 0.5:
                            #                 change = 0.6
                            #             change = (int(Stock_Change[len(Stock_Change)- 6]) + int(Stock_Change[len(Stock_Change)- 5]) + int(Stock_Change[len(Stock_Change)- 4]) + int(Stock_Change[len(Stock_Change)- 3]) + int(Stock_Change[len(Stock_Change)- 2]) +int(Stock_Change[len(Stock_Change)- 1]))/310
                            #             bs = "buy"
                            #         elif bs == "buy":
                            #             change = 0.35 * psar_boost
                            #             bs = "buy"
                            #         else:
                            #             change = 0.15* psar_boost
                            #             bs = "buy"
                            #     if (int(Stock_Change[len(Stock_Change)- 3]) + int(Stock_Change[len(Stock_Change)- 2]) +int(Stock_Change[len(Stock_Change)- 1])) < 0:
                            #         if change == 0:
                            #             if change < 0.5:
                            #                 change = 0.6
                            #             change = (int(Stock_Change[len(Stock_Change)- 4]) + int(Stock_Change[len(Stock_Change)- 3]) + int(Stock_Change[len(Stock_Change)- 2]) +int(Stock_Change[len(Stock_Change)- 1]))/110
                            #             bs = "sell"
                            #         elif bs == "sell":
                            #             change = 0.45
                            #             bs = "sell"
                            #         else:
                            #             change = 0.35
                            #             bs = "buy"

                                if predicted_outcome[0] == 1:
                                    bs = "buy"
                                    change = 1
                                if predicted_outcome_sell[0] == 1:
                                    bs = "sell"
                                    position = float(position)
                                    change = 1
                            #
                                position = float(position)
                                print(Cash, Stock, volatility)
                                precision_factor = 10000
                                bs = "sell"
                                if bs == "sell":
                                    units_to_ = floor(change* position * precision_factor)
                                    units_to_ /= precision_factor
                                    CST = CST

                                if bs == "buy":
                                    CST = ((Cash+Stock)-1700)/(volatility*100) + CST
                                    units_to_ = floor(change*cash_to_spend * precision_factor / latest_price)
                                    units_to_/= precision_factor


                                print('bs = ' + bs)
                                return (units_to_, change, bs, CST)
                            #
                            btc_units = []
                            btc_units.append(calculate_order_size(cash_to_spend, btc_latest_price, position, CST, Cash, Stock, volatility))
                            print(btc_units)
                            CSA = btc_units[0][3]
                            bs = btc_units[0][2]
                            change = btc_units[0][1]
                            btc_units = btc_units[0][0]

                            # print(bs)
                            print(change)
                            print(btc_units)
                            CST += CSA-CST

                            print(CST)


                            if bs == "buy":
                                print("Order Not Excecuted")

                            if bs == "sell":
                                print("Order Excecuted")
                                alpaca.submit_order(symbol=btc, qty=btc_units, side=bs, type='market', time_in_force='day')

                            print(bs + units_to_buy)
                        except:
                            print("")


                if b == 1:
                    if o == 1:
                        try:
                            btc = Stock_Name
                            Stock_Name = [btc]

                            account = alpaca.get_account()
                            account_cash = float(account.cash)
                            account_equity = float(account.equity)
                            try:
                                position = alpaca.get_position(btc).qty
                            except:
                                position = 0

                            cash_to_spend = ((account_cash-5000) / 1.1) * ((((Cash+Stock)-1700)/(volatility*100))/CST)

                            # btc_snapshot = alpaca.get_crypto_snapshot(symbol=btc, exchange="CBSE")
                            btc_snapshot = alpaca.get_snapshots(Stock_Name)
                            btc_latest_price = aaplc[len(aaplc)-1]
                            print(btc_latest_price)
                            # btc_latest_price = btc_snapshot.latest_quote.ap
                            bs = ""
                            def calculate_order_size(cash_to_spend, latest_price, position):
                                change = 0
                                psar_boost = 0.8
                                if psar_close[len(psar_close)-1] == aaplc[len(aaplc)-2]:
                                    psar_boost = 1.2
                                print(CS)

                                if CS < 1:
                                    change = 0.35 * psar_boost
                                    bs = "buy"
                                if CS > 1:
                                    change = 0.85
                                    bs = "sell"

                                if ( int(Stock_Change[len(Stock_Change)- 3]) + int(Stock_Change[len(Stock_Change)- 2]) + int(Stock_Change[len(Stock_Change)- 1])) > 0:
                                    if change == 0:
                                        if change < 0.5:
                                            change = 0.6
                                        change = (int(Stock_Change[len(Stock_Change)- 6]) + int(Stock_Change[len(Stock_Change)- 5]) + int(Stock_Change[len(Stock_Change)- 4]) + int(Stock_Change[len(Stock_Change)- 3]) + int(Stock_Change[len(Stock_Change)- 2]) +int(Stock_Change[len(Stock_Change)- 1]))/310
                                        bs = "buy"
                                    elif bs == "buy":
                                        change = 0.35 * psar_boost
                                        bs = "buy"
                                    else:
                                        change = 0.15* psar_boost
                                        bs = "buy"
                                if (int(Stock_Change[len(Stock_Change)- 3]) + int(Stock_Change[len(Stock_Change)- 2]) +int(Stock_Change[len(Stock_Change)- 1])) < 0:
                                    if change == 0:
                                        if change < 0.5:
                                            change = 0.6
                                        change = (int(Stock_Change[len(Stock_Change)- 4]) + int(Stock_Change[len(Stock_Change)- 3]) + int(Stock_Change[len(Stock_Change)- 2]) +int(Stock_Change[len(Stock_Change)- 1]))/110
                                        bs = "sell"
                                    elif bs == "sell":
                                        change = 0.45
                                        bs = "sell"
                                    else:
                                        change = 0.35
                                        bs = "buy"

                                position = float(position)

                                bs = "sell"
                                precision_factor = 10000
                                if bs == "sell":
                                    units_to_ = floor(change* position * precision_factor)
                                    units_to_ /= precision_factor
                                if bs == "buy":
                                    units_to_ = floor(change*cash_to_spend * precision_factor / latest_price)
                                    units_to_/= precision_factor
                                print('bs = ' + bs)
                                return (units_to_, change, bs)

                            btc_units = []
                            btc_units.append(calculate_order_size(cash_to_spend, btc_latest_price, position))
                            bs = btc_units[0][2]
                            change = btc_units[0][1]
                            btc_units = btc_units[0][0]

                            # print(bs)
                            print(change)
                            print(btc_units)
                            if bs == "sell":
                                print('Order Not Executed')
                            if bs == "buy":
                                print("Order Excecuted")
                                alpaca.submit_order(symbol=btc, qty=btc_units, side=bs, type='market', time_in_force='day')

                            print(bs + units_to_buy)
                        except:
                            print("")

    minutesToSleep = 1440 - (datetime.datetime.now().minute % 1440)
    time.sleep(minutesToSleep * 60)
    # minutesToSleep = 1440 - datetime.datetime.now().minute % 1440
    # time.sleep(minutesToSleep * 60)

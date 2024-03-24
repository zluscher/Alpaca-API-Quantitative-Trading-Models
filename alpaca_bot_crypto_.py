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



while True:

    for dirname, _, filenames in os.walk('/kaggle/input'):
        for filename in filenames:
            print(os.path.join(dirname, filename))


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

    # Stock_List = ['TQQQ','SQQQ','NRGU','NUGT','FNGU','FAS','LABU','SOXL','CURE','TMF','IYR']
    Stock_List= ['BTC-USD','LTC-USD','ETH-USD']




    import pytz
    aapl = pdr.get_data_yahoo(Stock_List[0],
                start=datetime.datetime.now(pytz.timezone('UTC')) + datetime.timedelta(-59),
                end=datetime.datetime.now(pytz.timezone('UTC')), interval="5m")

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


    for i in range(len(Stock_List)):
        Stock_Name = Stock_List[i]

        yfObj = yfin.Ticker(Stock_Name)
        data = yfObj.history(start='2022-1-1', end=datetime.date.today())
        indic = PSAR()

        aapl = pdr.get_data_yahoo(Stock_Name,
                          start=datetime.datetime.now(pytz.timezone('UTC')) + datetime.timedelta(-59),
                          end=datetime.datetime.now(pytz.timezone('UTC')), interval="5m")
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


    #     print(len(aaplc))
        # aaplo
        aapl['Log returns'] = np.log(aapl['Close']/aapl['Close'].shift())
        aapl['Log returns'].std()
        volatility = aapl['Log returns'].std()*252**.5

        minneg = []
        maxpos = []
        avgpos = 0
        avgneg = 0
        OBV = 0
        N = 0
        SMMA = 0
        OBVindicator = 0
        PeriodHigh = 0
        PeriodLow = 0
        diff_ = []

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
            OBVindicator = abs(OBVpre)-abs(OBV)

        minneg = min(maxpos)
        maxposv = max(maxpos)
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

        K__ = []
        OBV__ = []
        RSI__ = []
        N__ = []
        MACD__ = []

        def Indicator(value1, value2):
            minneg = []
            maxpos = []
            avgpos = 0
            avgneg = 0
            OBV = 0
            N = 0
            SMMA = 0
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

            minneg = min(maxpos)
            maxposv = max(maxpos)
            for i in range(len(maxpos)):
                if minneg == maxpos[i]:
                    PeriodHigh = i
                if maxpos == maxpos[i]:
                    PeriodLow = i
    #         Williams = (PeriodHigh-aaplc[i])/(PeriodHigh-PeriodLow)
            AroonUp = (-(PeriodHigh - length_time)/30)*100
            AroonDown = ((PeriodLow - length_time)/30)*100
            Aroon = (AroonUp + AroonDown)

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


            return (SMMA, OBV, N, RSI, K, Aroon, MACD, value1, value2)

        ##################################################################
        defaultbuy = [30,9,-0.5,-9,-50]
        defaultsell = [70,-20,0.5,-7,50]
        difference  = []
        Weight_List = []
        difference_W  = []
        Weight_List_W = []
        difference_M  = []
        Weight_List_M = []

        def Output():
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

            Rules(weight1, weight2, weight3, weight4, weight6, Rule_1_output, Rule_2_output, Rule_3_output, Rule_4_output, Rule_6_output)

            Rule_1S_output = Rule_1_S(RSI)[1]
            Rule_2S_output = Rule_2_S(K)[1]
            Rule_3S_output = Rule_3_S(N)[1]
            Rule_4S_output = Rule_4_S(OBV)[1]
            Rule_6S_output = Rule_6_S(MACD)[1]

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
                    if differenceM[i] > 0:
                        bullss += 1
                    else:
                        bearss += 1
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
                diffM = ((diffM/(len(difference_M)-1)))
                return (diffM, bullss, bearss)
        ##################################################################

        value2 = 14
        value4 = value2

        for i in range(16):
            aaplc.append(aaplc[len(aaplc)-1])
            aaplo.append(aaplo[len(aaplo)-1])
            aaplh.append(aaplh[len(aaplh)-1])
            aapll.append(aapll[len(aapll)-1])
            aaplv.append(aaplv[len(aaplv)-1])

        for i in range(len(stock)-value2):
            value1 = value2 + i

            if i >= len(stock)-value2-14:
                burned_variable = []
                burned_variable_W = []
                burned_variable.append(diff())
                burned_variable_W.append(diff_W())
                aaplc[i] += aaplc[i-1] * ((burned_variable[0][0] + burned_variable_W[0][0]/7)/2)/100
                aaplo[i] += aaplo[i-1] * ((burned_variable[0][0] + burned_variable_W[0][0]/7)/2)/100
                aaplh[i] += aaplh[i-1] * ((burned_variable[0][0] + burned_variable_W[0][0]/7)/2)/100
                aapll[i] += aapll[i-1] * ((burned_variable[0][0] + burned_variable_W[0][0]/7)/2)/100
                aaplv[i] += aaplv[i-1] * ((burned_variable[0][0] + burned_variable_W[0][0]/7)/2)/100
            RSI = Indicator(value1, value2)[3]
            K = Indicator(value1, value2)[4]
            OBV = Indicator(value1, value2)[1]
            N = Indicator(value1, value2)[2]
            SMMA = Indicator(value1, value2)[0]
            Aroon = Indicator(value1, value2)[5]
            MACD = Indicator(value1, value2)[6]
            Williams = Indicator(value1, value2)[7]
            K_.append(K)
            RSI_.append(RSI)
            OBV_.append(OBV)
            N_.append(N)
            SMMA_.append(SMMA)
            Aroon_.append(Aroon)
            MACD_.append(MACD)

            Output()


            if Buy[i] == 1:
                try:
                    K__.append(K_[i])
                    OBV__.append(OBV_[i])
                    RSI__.append(RSI_[i])
                    N__.append(N_[i])
                    MACD__.append(MACD_[i])
                except:
                    pass
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
        indic = PSAR()
        data['PSAR'] = data.apply(
            lambda x: indic.calcPSAR(x['High'], x['Low']), axis=1)
        data['EP'] = indic.ep_list
        data['Trend'] = indic.trend_list
        data['AF'] = indic.af_list
        data.head()
        indic._calcPSAR()

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

            for i in range(len(Buy)):
                f = int(i)
                if i == 0:
                    stock[f-1] = stock[f]

                Cash_.append(Cash)

                Change1 = -(stock[f-1]-stock[f])/(stock[f-1])

                Change = Stock *(-(stock[f-1]-stock[f])/(stock[f-1]))
                Stock += Stock * (-(stock[f-1]-stock[f])/(stock[f-1]))

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
                                Stock_Change.append('15')

                        if How_Much_Buy[j] == 1:
                            if Cash != 0 and Cash > 0:
                                Stock += Cash*value1
                                Cash -= Cash*value1
                                Stock_Change.append('35')

                        if How_Much_Buy[j] == 2:
                            if Cash != 0 and Cash > 0:
                                Stock += Cash*value2
                                Cash -= Cash*value2
                                Stock_Change.append('75')

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
                                Stock_Change.append('-15')

                        if How_Much_Sell[k] == 1:
                            if Stock != 0 and Stock > 0:
                                Cash += Stock*value2
                                Stock -= Stock*value2
                                Stock_Change.append('-35')

                        if How_Much_Sell[k] == 2:
                            if Stock != 0 and Stock > 0:
                                Cash += Stock*value3
                                Stock -= Stock*value3
                                Stock_Change.append('-75')

                        How_Much_Sell.append((How_Much_Sell[k] + 1))
                        k+=1
                if Buy[i] == 0 and Sell[i] == 0:
                    Stock_Change.append('0')
                day.append(i)
                account_dict["Stock"].append(Stock)
                account_dict["Cash"].append(Cash)
            Total = Stock + Cash
            return (Total, Stock_Change, Stock, Cash, Track_Change)
        List = []
        List.append(How_Much(Buy, Cash, Stock))
        Stock_Change = List[0][1]
        Stock_Exchange.append(Stock_Change)
        Stock = List[0][2]
        Cash = List[0][3]
        Stocks_Change.append(List[0][4])
        Stock_Names.append(Stock_Name)

        CS = Cash/Stock

    ################################################################
        print()
        print(Stock_Name)
        print("Cash/Stock Ratio " + str(CS)[:-13])
        print("You should Buy/Sell this percentage last 10 time periods " + Stock_Change[len(Stock_Change)- 6] + ' ' + Stock_Change[len(Stock_Change)- 5] +  ' ' + Stock_Change[len(Stock_Change)- 4] +  ' ' + Stock_Change[len(Stock_Change)- 3] + ' ' + Stock_Change[len(Stock_Change)- 2] + ' ' + Stock_Change[len(Stock_Change)- 1])

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

    ###################################
        btc = Stock_Name
        btc = btc[:-4] + "USD"
        print(btc)
        try:
            alpaca = api.REST('AK5AU7BPPHU7OL8BTZ34', 'okYooGCk3YgMSgtbCYQlYYNuqX6hri6noFbsbDz2', 'https://api.alpaca.markets')

            account = alpaca.get_account()
            account_cash = float(account.cash)
            account_equity = float(account.equity)
            position = alpaca.get_position(btc).qty
            cash_to_spend = account_cash / 1.1

            # btc_snapshot = alpaca.get_crypto_snapshot(symbol=btc, exchange="CBSE")
            btc_snapshot = alpaca.get_crypto_snapshot(symbol=btc, exchange="CBSE")
            btc_latest_price = btc_snapshot.latest_quote.ap
            # btc_latest_price = btc_snapshot.latest_quote.ap
            bs = ""

            def calculate_order_size(cash_to_spend, latest_price, position):
                change = 0
                if CS < 1:
                    change = 0.85
                    bs = "buy"
                if CS > 1:
                    change = 0.85
                    bs = "sell"

                if (int(Stock_Change[len(Stock_Change)- 6]) + int(Stock_Change[len(Stock_Change)- 5]) + int(Stock_Change[len(Stock_Change)- 4]) + int(Stock_Change[len(Stock_Change)- 3]) + int(Stock_Change[len(Stock_Change)- 2]) + int(Stock_Change[len(Stock_Change)- 1])) > 0:
                    if change == 0:
                        change = (int(Stock_Change[len(Stock_Change)- 6]) + int(Stock_Change[len(Stock_Change)- 5]) + int(Stock_Change[len(Stock_Change)- 4]) + int(Stock_Change[len(Stock_Change)- 3]) + int(Stock_Change[len(Stock_Change)- 2]) +int(Stock_Change[len(Stock_Change)- 1]))/110
                        bs = "buy"
                    elif bs == "buy":
                        change = 0.95
                        bs = "sell"
                    else:
                        change = 0.45
                        bs = "buy"

                if (int(Stock_Change[len(Stock_Change)- 6]) +int(Stock_Change[len(Stock_Change)- 5]) + int(Stock_Change[len(Stock_Change)- 4]) + int(Stock_Change[len(Stock_Change)- 3]) + int(Stock_Change[len(Stock_Change)- 2]) +int(Stock_Change[len(Stock_Change)- 1])) < 0:
                    if change == 0:
                        change = (int(Stock_Change[len(Stock_Change)- 6]) +int(Stock_Change[len(Stock_Change)- 5]) + int(Stock_Change[len(Stock_Change)- 4]) + int(Stock_Change[len(Stock_Change)- 3]) + int(Stock_Change[len(Stock_Change)- 2]) +int(Stock_Change[len(Stock_Change)- 1]))/110
                        bs = "sell"
                    elif bs == "sell":
                        change = 0.45
                        bs = "sell"
                    else:
                        change = 0.65
                        bs = "buy"

                position = float(position)

                precision_factor = 10000
                if bs == "sell":
                    units_to_ = floor(change* position * precision_factor)
                    units_to_ /= precision_factor
                if bs == "buy":
                    units_to_ = floor(change*cash_to_spend * precision_factor / latest_price)
                    units_to_/= precision_factor
                return (units_to_, change, bs)

            btc_units = []
            btc_units.append(calculate_order_size(cash_to_spend, btc_latest_price, position))
            # if (position/btc_latest_price) < 1:
            #     pass

            bs = btc_units[0][2]
            change = btc_units[0][1]
            btc_units = btc_units[0][0]
            print("Order Excecuted")
            alpaca.submit_order(symbol=btc, qty=btc_units, side=bs, type='market', time_in_force='gtc')

            print(bs + btc_units)

        except:
            print("")

    minutesToSleep = 5 - datetime.datetime.now().minute % 5
    time.sleep(minutesToSleep * 60)

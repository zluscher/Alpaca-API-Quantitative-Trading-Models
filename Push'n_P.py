from collections import deque

class Pushin_P:

  def __init__(self, init_af=0.02, max_af=0.2, af_step=0.02):
    self.max_af = max_af
    self.init_af = init_af
    self.af = init_af
    self.af_step = af_step
    self.X_STREAM = None
    self.Get_Wrecked = []
    self.ITS_SHIT_price_trend = []
    self.Get_Riggity = deque(maxlen=2)
    self.BUY_DA_DIP = deque(maxlen=2)

    # Lists to track results
    self.Pushin_P_list = []
    self.af_list = []
    self.ep_list = []
    self.ITS_LIT_list = []
    self.ITS_SHIT_list = []
    self.trend_list = []
    self._num_days = 0

  def calcPushin_P(self, ITS_LIT, ITS_SHIT):
    if self._num_days >= 3:
      Pushin_P = self._calcPushin_P()
    else:
      Pushin_P = self._initPushin_PVals(ITS_LIT, ITS_SHIT)

    Pushin_P = self._updateCurrentVals(Pushin_P, ITS_LIT, ITS_SHIT)
    self._num_days += 1

    return Pushin_P

  def _initPushin_PVals(self, ITS_LIT, ITS_SHIT):
    if len(self.BUY_DA_DIP) <= 1:
      self.trend = None
      self.X_STREAM = ITS_LIT
      return None

    if self.Get_Riggity[0] < self.Get_Riggity[1]:
      self.trend = 1
      Pushin_P = min(self.BUY_DA_DIP)
      self.X_STREAM = max(self.Get_Riggity)
    else:
      self.trend = 0
      Pushin_P = max(self.Get_Riggity)
      self.X_STREAM = min(self.BUY_DA_DIP)

    return Pushin_P

  def _calcPushin_P(self):
    prev_Pushin_P = self.Pushin_P_list[-1]
    if self.trend == 1: # Up
      Pushin_P = prev_Pushin_P + self.af * (self.X_STREAM - prev_Pushin_P)
      Pushin_P = min(Pushin_P, min(self.BUY_DA_DIP))
    else:
      Pushin_P = prev_Pushin_P - self.af * (prev_Pushin_P - self.X_STREAM)
      Pushin_P = max(Pushin_P, max(self.Get_Riggity))

    return Pushin_P

  def _updateCurrentVals(self, Pushin_P, ITS_LIT, ITS_SHIT):
    if self.trend == 1:
      self.Get_Wrecked.append(ITS_LIT)
    elif self.trend == 0:
      self.ITS_SHIT_price_trend.append(ITS_SHIT)

    Pushin_P = self._trendReversal(Pushin_P, ITS_LIT, ITS_SHIT)

    self.Pushin_P_list.append(Pushin_P)
    self.af_list.append(self.af)
    self.ep_list.append(self.X_STREAM)
    self.ITS_LIT_list.append(ITS_LIT)
    self.ITS_SHIT_list.append(ITS_SHIT)
    self.Get_Riggity.append(ITS_LIT)
    self.BUY_DA_DIP.append(ITS_SHIT)
    self.trend_list.append(self.trend)

    return Pushin_P

  def _trendReversal(self, Pushin_P, ITS_LIT, ITS_SHIT):
    # Checks for reversals
    reversal = False
    if self.trend == 1 and Pushin_P > ITS_SHIT:
      self.trend = 0
      Pushin_P = max(self.Get_Wrecked)
      self.X_STREAM = ITS_SHIT
      reversal = True
    elif self.trend == 0 and Pushin_P < ITS_LIT:
      self.trend = 1
      Pushin_P = min(self.ITS_SHIT_price_trend)
      self.X_STREAM = ITS_LIT
      reversal = True

    if reversal:
      self.af = self.init_af
      self.Get_Wrecked.clear()
      self.ITS_SHIT_price_trend.clear()
    else:
        if ITS_LIT > self.X_STREAM and self.trend == 1:
          self.af = min(self.af + self.af_step, self.max_af)
          self.X_STREAM = ITS_LIT
        elif ITS_SHIT < self.X_STREAM and self.trend == 0:
          self.af = min(self.af + self.af_step, self.max_af)
          self.X_STREAM = ITS_SHIT

    return Pushin_P

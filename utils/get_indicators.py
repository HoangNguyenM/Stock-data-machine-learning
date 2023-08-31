import ta

"""Define functions for each indicator"""

# Calculate RSI (Relative Strength Index)
def get_rsi(close):
    return ta.momentum.RSIIndicator(close).rsi()

# Calculate MACD (Moving Average Convergence Divergence)
def get_macd(close, window_slow = 26, window_fast = 12, window_sign = 9):
    macd = ta.trend.MACD(close, window_slow=window_slow, window_fast=window_fast, window_sign=window_sign)
    return macd.macd(), macd.macd_signal()

# Calculate EMAs (Exponential Moving Averages)
def get_ema(close, window):
    return ta.trend.EMAIndicator(close, window=window).ema_indicator()

# Calculate VWAP (Volume Weighted Average Price)
def get_vwap(high, low, close, volume):
    vwap_indicator = ta.volume.VolumeWeightedAveragePrice(high=high, low=low, close=close, volume=volume)
    return vwap_indicator.vwap

# Calculate Bollinger Bands, return upper, lower, middle bands
def get_bollinger_bands(close, window = 20, dev = 2.4):
    bollinger_bands = ta.volatility.BollingerBands(close=close, window=window, window_dev=dev)
    return bollinger_bands.bollinger_hband(), bollinger_bands.bollinger_lband(), bollinger_bands.bollinger_mavg()

# Calculate Donchian Channel over a specific period, return upper, lower, middle channels
def get_donchian_channel(high, low, period):
    upper = high.rolling(30).max().shift(1)  # Shifted by one period
    lower = low.rolling(30).min().shift(1)   # Shifted by one period
    return upper, lower, (upper + lower) / 2

# Calculate Average True Range (ATR)
def get_atr(high, low, close):
    atr_indicator = ta.volatility.AverageTrueRange(high=high, low=low, close=close)
    return atr_indicator.average_true_range()

# Calculate True Strength Index (TSI)
def get_tsi(close):
    tsi_indicator = ta.momentum.TSIIndicator(close=close)
    return tsi_indicator.tsi()
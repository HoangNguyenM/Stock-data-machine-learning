from abc import ABC, abstractmethod
import numpy as np

from keras.models import load_model

from models import CNN
from utils import get_indicators

class Signals(ABC):
    """RSI conditions"""
    def RSIoverSold(self, RSIvalue):
        return get_indicators.get_rsi(self.data["Close"]) < RSIvalue
    
    def RSIoverBought(self, RSIvalue):
        return get_indicators.get_rsi(self.data["Close"]) > RSIvalue
    
    """MACD conditions"""
    def MACDcrossover(self, period_ago = 0):
        macd, macd_signal = get_indicators.get_macd(self.data["Close"])
        return (macd.shift(period_ago) > macd_signal.shift(period_ago)) \
            & (macd.shift(period_ago + 1) <= macd_signal.shift(period_ago + 1)) \
            & (macd_signal.shift(period_ago) < 0)

    def MACDcrossunder(self, period_ago = 0):
        macd, macd_signal = get_indicators.get_macd(self.data["Close"])
        return (macd.shift(period_ago) < macd_signal.shift(period_ago)) \
            & (macd.shift(period_ago + 1) >= macd_signal.shift(period_ago + 1)) \
            & (macd_signal.shift(period_ago) > 0)
    
    # return if MACD cross over in recent periods, including the current
    def recentMACDcrossover(self, periods = 1):
        # create Signal that contains all False, does not change the output of "OR"
        Signal = self.data["Close"] != self.data["Close"]
        for period in range(0, periods):
            Signal = Signal | self.MACDcrossover(period_ago = period)
        return Signal
    
    # return if MACD cross under in recent periods, including the current
    def recentMACDcrossunder(self, periods = 1):
        # create Signal that contains all False, does not change the output of "OR"
        Signal = self.data["Close"] != self.data["Close"]
        for period in range(0, periods):
            Signal = Signal | self.MACDcrossunder(period_ago = period)
        return Signal

    # to set a threshold, MACD needs to be normalized for all stocks
    def scaledMACDcrossover(self, threshold = -1):
        macd, macd_signal = get_indicators.get_macd(self.data["Close"])
        ema200 = get_indicators.get_ema(self.data["Close"], 200)
        MACD = macd / ema200 * 1000
        Signal = macd_signal / ema200 * 1000
        return (MACD > Signal) & (MACD.shift(1) <= Signal.shift(1)) & (Signal < threshold)
    
    def scaledMACDcrossunder(self, threshold = 1):
        macd, macd_signal = get_indicators.get_macd(self.data["Close"])
        ema200 = get_indicators.get_ema(self.data["Close"], 200)
        MACD = macd / ema200 * 1000
        Signal = macd_signal / ema200 * 1000
        return (MACD < Signal) & (MACD.shift(1) >= Signal.shift(1)) & (Signal > threshold)
    
    def MACDvolHigh(self, ratio):
        macd_vol, macd_vol_signal = get_indicators.get_macd(self.data["Volume"])
        return macd_vol > (macd_vol_signal * ratio)

    """EMA conditions"""
    def EMAcrossover(self, emashort, emalong, period_ago = 0):
        return (emashort.shift(period_ago) > emalong.shift(period_ago)) \
            & (emashort.shift(period_ago + 1) <= emalong.shift(period_ago + 1))
    
    def EMAcrossunder(self, emashort, emalong, period_ago = 0):
        return (emashort.shift(period_ago) < emalong.shift(period_ago)) \
            & (emashort.shift(period_ago + 1) >= emalong.shift(period_ago + 1))
    
    # return if EMA cross over in recent periods, including the current
    def recentEMAcrossover(self, EMAshort = 10, EMAlong = 30, periods = 1):
        emashort = get_indicators.get_ema(self.data["Close"], EMAshort)
        emalong = get_indicators.get_ema(self.data["Close"], EMAlong)
        # create Signal that contains all False, does not change the output of "OR"
        Signal = self.data["Close"] != self.data["Close"]
        for period in range(0, periods):
            Signal = Signal | self.EMAcrossover(emashort, emalong, period_ago = period)
        return Signal
    
    # return if EMA cross under in recent periods, including the current
    def recentEMAcrossunder(self, EMAshort = 10, EMAlong = 30, periods = 1):
        emashort = get_indicators.get_ema(self.data["Close"], EMAshort)
        emalong = get_indicators.get_ema(self.data["Close"], EMAlong)
        # create Signal that contains all False, does not change the output of "OR"
        Signal = self.data["Close"] != self.data["Close"]
        for period in range(0, periods):
            Signal = Signal | self.EMAcrossunder(emashort, emalong, period_ago = period)
        return Signal
    
    """VWAP conditions"""
    def aboveVWAP(self, ratio):
        vwap = get_indicators.get_vwap(high=self.data["High"], low=self.data["Low"], close=self.data["Close"], volume=self.data["Volume"])
        return self.data["Close"] > (vwap * ratio)
    
    def belowVWAP(self, ratio):
        vwap = get_indicators.get_vwap(high=self.data["High"], low=self.data["Low"], close=self.data["Close"], volume=self.data["Volume"])
        return self.data["Close"] < (vwap * ratio)
    
    """Donchian Channels conditions"""
    def DonchianBreakout(self, Donperiod = 30, consecutive_periods = 2):
        DonchianUpper, _, _ = get_indicators.get_donchian_channel(high=self.data["High"], low=self.data["Low"], period=Donperiod)
        # create Signal that contains all True, does not change the output of "AND"
        Signal = self.data["Close"] == self.data["Close"]
        for period in range(0, consecutive_periods):
            Signal = Signal & (self.data["Close"].shift(period) >= DonchianUpper.shift(period))
        return Signal
    
    def DonchianBreakUnder(self, Donperiod = 30, consecutive_periods = 2):
        _, DonchianLower, _ = get_indicators.get_donchian_channel(high=self.data["High"], low=self.data["Low"], period=Donperiod)
        # create Signal that contains all True, does not change the output of "AND"
        Signal = self.data["Close"] == self.data["Close"]
        for period in range(0, consecutive_periods):
            Signal = Signal & (self.data["Close"].shift(period) <= DonchianLower.shift(period))
        return Signal
    
    """CNN prediction conditions"""
    def CNNpredHigh(self, prior_duration=60, post_duration=30, threshold=0):
        return self.data[f'CNN_{prior_duration}_{post_duration}'] > threshold
    
    def CNNpredLow(self, prior_duration=60, post_duration=30, threshold=0):
        return self.data[f'CNN_{prior_duration}_{post_duration}'] < threshold

class Strategy(Signals):
    def __init__(self, data):
        self.data = data

    @abstractmethod
    def entry_signal(self):
        pass

    @abstractmethod
    def exit_signal(self):
        pass

    @abstractmethod
    def run_strategy(self):
        pass

    def exit_by_ATR(self, entry_signals, profit_ratio=4, loss_ratio=2, **kwargs):
        long_entry_indices = self.data.index[entry_signals == 1].to_numpy()
        short_entry_indices = self.data.index[entry_signals == -1].to_numpy()
        # if a signal is given at the last data point, remove
        if len(long_entry_indices) > 0 and long_entry_indices[-1] == self.data.index[-1]:
            long_entry_indices = long_entry_indices[:-1]
        if len(short_entry_indices) > 0 and short_entry_indices[-1] == self.data.index[-1]:
            short_entry_indices = short_entry_indices[:-1]

        long_trades = []
        short_trades = []
        
        curr_index = 0

        if len(long_entry_indices) > 0 or len(short_entry_indices) > 0:
            # get ATR
            ATR = get_indicators.get_atr(high=self.data["High"], low=self.data["Low"], close=self.data["Close"])

        while len(long_entry_indices) > 0 or len(short_entry_indices) > 0:
            # find the next possible signal for a trade
            if len(short_entry_indices) == 0:
                curr_index = long_entry_indices[0]
                position = "long"
            elif len(long_entry_indices) == 0:
                curr_index = short_entry_indices[0]
                position = "short"
            elif long_entry_indices[0] <= short_entry_indices[0]:
                curr_index = long_entry_indices[0]
                position = "long"
            else:
                curr_index = short_entry_indices[0]
                position = "short"

            atr_entry = ATR.loc[curr_index]

            if len(self.data["Close"]) - curr_index > self.maxlen:
                exit_index = self.data.index[(curr_index + 1):(curr_index + self.maxlen)]
            else:
                exit_index = self.data.index[curr_index + 1:]

            # exit if the profit is 2 * ATR (at entry) or the loss is 1 * ATR (at entry)
            if position == "long":
                exit_index_valid = exit_index[(self.data["Close"].loc[exit_index] <= (self.data["Close"].loc[curr_index] - loss_ratio * atr_entry)) \
                                    | (self.data["Close"].loc[exit_index] >= (self.data["Close"].loc[curr_index] + profit_ratio * atr_entry))]
            else:
                exit_index_valid = exit_index[(self.data["Close"].loc[exit_index] >= (self.data["Close"].loc[curr_index] + loss_ratio * atr_entry)) \
                                    | (self.data["Close"].loc[exit_index] <= (self.data["Close"].loc[curr_index] - profit_ratio * atr_entry))]

            if len(exit_index_valid) == 0:
                exit_index = exit_index[-1]
            else:
                exit_index = exit_index_valid[0]
            
            # print([curr_index, exit_index])
            # record one trade
            if position == "long":
                long_trades.append([curr_index, exit_index])
            else:
                short_trades.append([curr_index, exit_index])
            
            curr_index = exit_index

            # remove the missed opportunities
            long_entry_indices = long_entry_indices[long_entry_indices >= curr_index]

            short_entry_indices = short_entry_indices[short_entry_indices >= curr_index]
        
        long_trades = np.array(long_trades).reshape((-1,2))
        short_trades = np.array(short_trades).reshape((-1,2))
        return long_trades[:,0], long_trades[:,1], short_trades[:,0], short_trades[:,1]


    def exit_by_ATR_w_overlap(self, entry_signals, profit_ratio=4, loss_ratio=2, **kwargs):
        # get long entry indices
        long_entry_indices = self.data.index[entry_signals == 1].to_numpy()
        # get short entry indices
        short_entry_indices = self.data.index[entry_signals == -1].to_numpy()
        # get ATR
        if len(long_entry_indices) > 0 or len(short_entry_indices) > 0:
            ATR = get_indicators.get_atr(high=self.data["High"], low=self.data["Low"], close=self.data["Close"])

        # create a function to calculate exit indices given entry indices and the position type
        def get_exit_signals(entry_indices, position = 'long'):
            # if a signal is given at the last data point, remove
            if entry_indices[-1] == self.data.index[-1]:
                entry_indices = entry_indices[:-1]

            # get entry prices
            if position == 'long':
                profit_exit = ATR[entry_indices] * profit_ratio + self.data["Close"][entry_indices]
                loss_exit = - ATR[entry_indices] * loss_ratio + self.data["Close"][entry_indices]
                
                price_windows = np.full((len(entry_indices), self.maxlen), 1e5)
                indices_windows = np.full((len(entry_indices), self.maxlen), self.data.index[-1])
            else:
                profit_exit = - ATR[entry_indices] * profit_ratio + self.data["Close"][entry_indices]
                loss_exit = ATR[entry_indices] * loss_ratio + self.data["Close"][entry_indices]
                
                price_windows = np.full((len(entry_indices), self.maxlen), 0)
                indices_windows = np.full((len(entry_indices), self.maxlen), self.data.index[-1])

            # input prices for windows of trades
            for i in range(len(entry_indices)):
                curr_index = entry_indices[i]
                if self.data.index[-1] + 1 - curr_index > self.maxlen:
                    price_windows[i, :(self.maxlen - 1)] = self.data["Close"][(curr_index + 1):(curr_index + self.maxlen)]
                    indices_windows[i, :(self.maxlen - 1)] = self.data.index[(curr_index + 1):(curr_index + self.maxlen)]
                else:
                    price_windows[i, :(self.data.index[-1]-curr_index)] = self.data["Close"][curr_index + 1:]
                    indices_windows[i, :(self.data.index[-1]-curr_index)] = self.data.index[curr_index + 1:]
            
            profit_exit = np.array(profit_exit)[:, np.newaxis]
            profit_exit = np.tile(profit_exit, (1, self.maxlen))

            # calculate indices for profit exits
            if position == 'long':
                profit_indices_values = np.argmax(price_windows >= profit_exit, axis=1)
            else:
                profit_indices_values = np.argmax(price_windows <= profit_exit, axis=1)

            profit_exit_indices = indices_windows[np.arange(len(entry_indices)), profit_indices_values]

            if position == 'long':
                price_windows[price_windows == 1e5] = 0
            else:
                price_windows[price_windows == 0] = 1e5

            loss_exit = np.array(loss_exit)[:, np.newaxis]
            loss_exit = np.tile(loss_exit, (1, self.maxlen))

            # calculate indices for loss exits
            if position == 'long':
                loss_indices_values = np.argmax(price_windows <= loss_exit, axis=1)
            else:
                loss_indices_values = np.argmax(price_windows >= loss_exit, axis=1)

            loss_exit_indices = indices_windows[np.arange(len(entry_indices)), loss_indices_values]

            return np.minimum(profit_exit_indices, loss_exit_indices)
        
        """Get long exits"""
        if len(long_entry_indices) > 0:
            long_exit_indices = get_exit_signals(long_entry_indices, 'long')
        else:
            long_exit_indices = []

        """Get short exits"""
        if len(short_entry_indices) > 0:
            short_exit_indices = get_exit_signals(short_entry_indices, 'short')
        else:
            short_exit_indices = []

        return long_entry_indices, long_exit_indices, short_entry_indices, short_exit_indices
    
    # Use the function below for training process
    def calculate_profit(self, starting_balance=1, **kwargs):
        long_entry, long_exit, short_entry, short_exit = self.run_strategy(**kwargs)
        
        balance = starting_balance

        if long_entry.shape[0] > 0:
            long_exit_prices = self.data["Close"].values[long_exit]
            long_entry_prices = self.data["Close"].values[long_entry]
            balance *= np.prod(long_exit_prices / long_entry_prices)

        if short_entry.shape[0] > 0:
            short_exit_prices = self.data["Close"].values[short_exit]
            short_entry_prices = self.data["Close"].values[short_entry]
            balance *= np.prod(short_entry_prices / short_exit_prices)

        return balance
    
    # Use the function below for testing process, also print out some information
    def calculate_profit_w_verbal(self, starting_balance=1, **kwargs):
        long_entry, long_exit, short_entry, short_exit = self.run_strategy(**kwargs)

        balance = starting_balance
        
        if long_entry.shape[0] > 0:
            long_exit_prices = self.data["Close"].values[long_exit]
            long_entry_prices = self.data["Close"].values[long_entry]
            balance *= np.prod(long_exit_prices / long_entry_prices)
            long_len = np.mean(np.array(long_exit) - np.array(long_entry))
        else:
            long_exit_prices = 1
            long_entry_prices = 1
            long_len = 0

        if short_entry.shape[0] > 0:
            short_exit_prices = self.data["Close"].values[short_exit]
            short_entry_prices = self.data["Close"].values[short_entry]
            balance *= np.prod(short_entry_prices / short_exit_prices)
            short_len = np.mean(np.array(short_exit) - np.array(short_entry))
        else:
            short_exit_prices = 1
            short_entry_prices = 1
            short_len = 0
        
        return balance, len(long_entry), long_len, len(short_entry), short_len, \
            np.prod(long_exit_prices / long_entry_prices), np.prod(short_entry_prices / short_exit_prices), \
            self.data["Close"].iloc[-1] / self.data["Close"].iloc[0]
    
class Strategy_Indicators(Strategy):
    '''Using Indicators only:
    Entry: EMA crossover, MACD crossover, RSI, VWAP, Donchian breakout
    Exit: ATR based profit/loss
    '''
    def __init__(self, data, short=True):
        super().__init__(data)
        self.name = 'Indicators'
        self.short = short
        self.maxlen = 1000
        
    def run_strategy(self, **kwargs):
        """Args:
        EMAshort: duration for short EMA
        EMAlong: duration for long EMA
        Donperiod: duration for Donchian channels
        RSIoversold: if RSI is lower than this level -> buy signal
        VWAPratio: VWAP * this ratio determines the uptrend or downtrend (above or below stock price)
        EMAperiod: check if EMA cross over/under in recent number of periods
        MACDperiod: check if MACD cross over/under in recent number of periods
        consecutiveBreak: number of consecutive periods the price break resistance/support
        Conditionsnum: number of primary conditions that needs fulfilled to create signal

        profit_ratio: ATR * this ratio for stop profit
        loss_ratio: ATR * this ratio for stop loss
        """
        # make entry signals
        entry_signals = self.entry_signal(**kwargs)

        # return entry and exit points
        return self.exit_signal(entry_signals, **kwargs)

    def entry_signal(self, RSIoverSold=90, VWAPratio=0.5, 
                     EMAshort=10, EMAlong=30, EMAperiod=5, MACDperiod=5, 
                     Donperiod=30, consecutiveBreak=2, 
                     Conditionsnum=2, **kwargs):
        
        # return 1 for long position and -1 for short position signals
        entry_signals = np.zeros(len(self.data))
        
        # make entry signals for long positions
        long_primary_conditions = self.recentEMAcrossover(EMAshort=EMAshort, EMAlong=EMAlong, periods=EMAperiod).astype(int) \
            + self.recentMACDcrossover(periods=MACDperiod).astype(int) \
            + self.DonchianBreakout(Donperiod=Donperiod, consecutive_periods=consecutiveBreak).astype(int)
        long_conditions = (long_primary_conditions >= Conditionsnum) \
            & self.RSIoverSold(RSIvalue=RSIoverSold) & self.aboveVWAP(ratio=VWAPratio)
        entry_signals[long_conditions] = 1

        # make entry signals for short positions
        if self.short:
            short_primary_conditions = self.recentEMAcrossunder(EMAshort=EMAshort, EMAlong=EMAlong, periods=EMAperiod).astype(int) \
                + self.recentMACDcrossunder(periods=MACDperiod).astype(int) \
                + self.DonchianBreakUnder(Donperiod=Donperiod, consecutive_periods=consecutiveBreak).astype(int)
            short_conditions = (short_primary_conditions >= Conditionsnum) \
                & self.RSIoverBought(RSIvalue=100-RSIoverSold) & self.belowVWAP(ratio=1/VWAPratio)
            entry_signals[short_conditions] = -1

        return entry_signals

    def exit_signal(self, entry_signals, **kwargs):
        # exit using ATR
        return self.exit_by_ATR(entry_signals, **kwargs)


class Strategy_CNN_ATR(Strategy):
    '''CNN Signal as primary, combined with Indicators:
    Entry: CNN output and EMA crossover, MACD crossover, RSI, VWAP, Donchian breakout
    Exit: ATR based profit/loss
    
    '''
    def __init__(self, data, short=True):
        super().__init__(data)
        self.name = 'CNN'
        self.short = short
        self.maxlen = 1000

    def run_strategy(self, **kwargs):
        # make entry signals
        entry_signals = self.entry_signal(prior_duration=kwargs['priorDuration'], post_duration=kwargs['postDuration'],
                                          CNNHigh=kwargs['CNNHigh'],
                                          RSIoverSold=kwargs['RSIoverSold'], VWAPratio=kwargs['VWAPratio'])

        # return entry and exit points
        return self.exit_signal(entry_signals, profit_ratio=kwargs['profit_ratio'], loss_ratio=kwargs['loss_ratio'])

    def entry_signal(self, prior_duration=60, post_duration=30, CNNHigh=0, RSIoverSold=50, VWAPratio=1):
        # return 1 for long position and -1 for short position signals
        entry_signals = np.zeros(len(self.data))
        
        # make entry signals for long positions
        long_conditions = self.CNNpredHigh(prior_duration=prior_duration, post_duration=post_duration, threshold=CNNHigh) \
            & (self.EMAcrossover() | self.MACDcrossover()) \
            & self.RSIoverSold(RSIvalue=RSIoverSold) & self.aboveVWAP(ratio=VWAPratio)
        entry_signals[long_conditions] = 1

        # make entry signals for short positions
        if self.short:
            short_conditions = self.CNNpredLow(prior_duration=prior_duration, post_duration=post_duration, threshold=-CNNHigh) \
                & (self.EMAcrossunder() | self.MACDcrossunder()) \
                & self.RSIoverBought(RSIvalue=100-RSIoverSold) & self.belowVWAP(ratio=1/VWAPratio)
            entry_signals[short_conditions] = -1

        return entry_signals

    def exit_signal(self, entry_signals, profit_ratio=2, loss_ratio=1):
        # exit using ATR
        return self.exit_by_ATR(entry_signals, profit_ratio, loss_ratio)


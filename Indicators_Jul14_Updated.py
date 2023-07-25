class Indicators():
    def __init__(self) -> None:
        pass
        
    def create_dataframe(self):
        """Create dataframe and plot stck prices from a given stock

        Args:
            outlier_cutoff (int): Used for remove data outside the outlier (Unused in this case)

        Returns:
            tickers; test_stock: Stock codes and prices for all stocks
        """
        
        # Read datasets
        sp500_daily = pd.read_feather('FCF Holdings Daily 2018-06-30.feather')
        
        # Drop null values
        sp500_daily.dropna(how='any',inplace=True)
        
        # Convert time into datetime
        sp500_daily['date'] = pd.to_datetime(sp500_daily['date'])
        
        # Get stock codes for all stocks
        tickers = sp500_daily.ticker.unique()
        sp500_daily = sp500_daily.set_index(['ticker','date']).sort_index()
        sp500_daily.groupby('ticker').size().unique()
        
        return tickers, sp500_daily
    
    def create_test_stock(self, ticker, sp500_daily, dailyOrWeekly):
        """Create test stocks for daily or weekly research

        Args:
            ticker (str): a stock's ticker
            sp500_daily (dataframe): all stocks info
            dailyOrWeekly (str): choose Daily or Weekly mode

        Returns:
            Daily or Weekly data depends on the variable dailyOrWeekly
        """
        
        test_stock_daily = sp500_daily.loc[ticker,:]
        if dailyOrWeekly == 'Weekly':
            r = test_stock_daily[['close', 'high', 'low', 'volume']].resample('W')
            r_agg = r.agg(['last', 'max', 'min'])
            close_weekly, high_weekly, low_weekly, volume_weekly, index_weekly = r_agg['close']['last'], r_agg['high']['max'], r_agg['low']['min'], r_agg['volume']['last'], r_agg.index
            Frame = {'close': close_weekly,
                    'high': high_weekly,
                    'low': low_weekly,
                    'volume': volume_weekly}
            test_stock_weekly = pd.DataFrame(Frame)
        if dailyOrWeekly == 'Daily':
            test_stock = test_stock_daily  
        else: 
            test_stock = test_stock_weekly
        
        return test_stock, len(test_stock)
    
    # Build Indicators
    def RSI_BB_MACD(self, test_price):
        """Build RSI/ Bollinger Bands/ MACD indicators

        Args:
            test_price (dataframe): stock prices for a give stock

        Returns:
            values for technical indicators (RSI/ BB/ MACD)
        """
        
        rsi = RSI(test_price, timeperiod=14)
        up, mid, low = BBANDS(test_price, timeperiod=21, nbdevup=2, nbdevdn=2, matype=0)
        macd, macdsignal, macdhist = MACD(test_price, fastperiod=12, slowperiod=26, signalperiod=9)
        data = pd.DataFrame({'Test Price': test_price, 'BB Up': up, 'BB Mid': mid, 'BB down': low, 'RSI': rsi, 'MACD': macd})
        return rsi, macd, macdsignal, macdhist, up, mid, low, data

    def plot_RSI_BB_MACD(self, test_price, rsi, macd, macdsignal, macdhist, up, mid, low,  data):
        """Plot technical indicators(RSI/ BB/ MACD)

        Args:
            test_price (dataframe): stock prices for a give stock
            rsi (dataframe): 
            macd (dataframe): 
            macdsignal (dataframe): 
            macdhist(dataframe): 
            up (dataframe): 
            mid (dataframe): 
            low (dataframe): 
            data (dataframe): 
        """
        
        macd_data = pd.DataFrame({'Test Price': test_price, 'MACD': macd, 'MACD Signal': macdsignal, 'MACD History': macdhist})
        fig, axes= plt.subplots(nrows=2, figsize=(15, 8))
        macd_data['Test Price'].plot(ax=axes[0])
        macd_data.drop('Test Price', axis=1).plot(ax=axes[1])
        fig.tight_layout()
        sns.despine();
        
        fig, axes= plt.subplots(nrows=3, figsize=(15, 10), sharex=True)
        data.drop(['RSI', 'MACD'], axis=1).plot(ax=axes[0], lw=1, title='Bollinger Bands')
        data['RSI'].plot(ax=axes[1], lw=1, title='Relative Strength Index')
        axes[1].axhline(70, lw=1, ls='--', c='k')
        axes[1].axhline(30, lw=1, ls='--', c='k')
        data.MACD.plot(ax=axes[2], lw=1, title='Moving Average Convergence/Divergence', rot=0)
        axes[2].set_xlabel('')
        fig.tight_layout()
        sns.despine();

    # calculate Hull Moving Average (Hull MA)
    def hull_ma(self, src, length):
        """Compute Hull Moving Average

        Args:
            test_price (DataFrame): only stock prices with index Periods_date
            length (int): window length

        Returns:
            WMA: Hull Moving Average
        """
            
        wma1 = 2 * WMA(src, int(length/2))
        wma2 = WMA(src, length)
        diff = wma1 - wma2
        sqrtLength = round(np.sqrt(length))
        return WMA(diff, sqrtLength)

    # calculate Triple Exponential Moving Average (TEMA)
    def t_ema(self, src, length):
        """Compute Triple Exponential Moving Average

            Args:
                test_price (DataFrame): only stock prices with index Periods_date
                length (int): window length

            Returns:
                WMA: Triple Exponential Moving Average
        """
        
        ema1 = src.ewm(span=length, adjust=False).mean()
        ema2 = ema1.ewm(span=length, adjust=False).mean()
        ema3 = ema2.ewm(span=length, adjust=False).mean()
        return 3 * (ema1 - ema2) + ema3

    # calculate Tilson T3
    def tilson_t3(self, src, length, factor):
        """Compute Tillson T3 Moving Average

            Args:
                test_price (DataFrame): only stock prices with index Periods_date
                length (int): window length

            Returns:
                WMA: Tillson T3 Moving Average
        """
        
        gd = lambda x: x.ewm(span=length, adjust=False).mean() * (1 + factor) - x.ewm(span=length, adjust=False).mean().ewm(span=length, adjust=False).mean() * factor
        t3 = lambda x: gd(gd(gd(x)))
        return t3(src)

    def VWMA(self, df, length):
        """Plot 3 moving averages for visualization

            Args:
                test_price (DataFrame): only stock prices with index Periods_date
        """
            
        pivot = (df.high + df.low + df.close)/3 * df.volume
        VWMA = pivot.rolling(length).mean()/df.volume.rolling(length).mean()
        return VWMA

    def compute_moving_average(self, src, df, len1, atype, factorT3):
        """Build and compute 8 moving averages as below:
        1: Simple Moving Average (SMA)
        2: Exponential Moving Average (Not Adjusted)
        3: Weighted Moving Average (WMA)
        4: Hull Moving Average (Hull MA)
        5: Volume Weighted Moving Average (VWMA)
        6: Exponential Moving Average (Adjusted)
        7: Triple Exponential Moving Average (TEMA)
        8: Tilson T3 Moving Average (Tilson T3)

        Args:
            src (DataFrame): contains all info about stocks
            df (DataFrame): contains all info about stocks
            doma2 (bollean): whether do the second moving average
            len1 (int): length of the first window
            len2 (int): length of the second window
            atype (int): type of the first moving average
            atype2 (int): type of the second moving average

        Returns:
            src: moving averages for a give type MA
        """
        
        hullma1 = self.hull_ma(src, len1)
        ema1 = src.ewm(span=len1, adjust=False).mean()
        ema2 = ema1.ewm(span=len1, adjust=False).mean()
        ema3 = ema2.ewm(span=len1, adjust=False).mean()
        tema = self.t_ema(src, len1)
        tilT3 = self.tilson_t3(src, len1, factorT3)
        vwma = self.VWMA(df, len1)

        def compute_avg(src, atype, hullma1, ema1, tilT3, vwma, len1):
            avg = np.where(atype == 1, pd.Series(src).rolling(window=len1).mean(),
                        np.where(atype == 2, ema1,
                                    np.where(atype == 3, src.rolling(window=len1).apply(lambda x: np.dot(x, np.arange(len1)) / np.sum(np.arange(len1))),
                                            np.where(atype == 4, hullma1,
                                                        np.where(atype == 5, vwma,
                                                                np.where(atype == 6, src.ewm(span=len1, adjust=True).mean(),
                                                                        np.where(atype == 7, tema,
                                                                                np.where(atype == 8, tilT3, None))))))))
            return avg
        
        out1 = compute_avg(src, atype, hullma1, ema1, tilT3, vwma, len1)
            
        src = pd.Series(src)
        out1 = pd.Series(out1.astype(float),index=df.index)

        return src, out1

    def buy_sell_signals_MA(self, signals, src, out1, atype_name):
        """Build buy and sell signals for moving averages

        Args:
            signals (dataframe): stock prices for a dataframe
            src (dataframe): moving averages for a give type MA
            out1 (dataframe): moving averages for a give type MA
            atype_name (str): the name of a MA
            doma2 (_type_): _description_

        Returns:
            buy_dates, buy_price, sell_dates, sell_price, positions
        """
        
        # Signal when price crosses the first moving average
        signals[atype_name + ' Buy'] = np.where((src.shift(1) < out1.shift(1)) & (src > out1), 1, 0)
        buy_dates = signals.loc[signals[atype_name + ' Buy'] == 1].index
        buy_price = src[signals[atype_name + ' Buy'] == 1]
        signals[atype_name + ' Sell'] = np.where((src.shift(1) > out1.shift(1)) & (src < out1), -1, 0)
        sell_dates = signals.loc[signals[atype_name + ' Sell'] == -1].index
        sell_price = src[signals[atype_name + ' Sell'] == -1]
        
        # Calculate positions based on the trading signals
        positions = signals['Signal'].diff()
        return buy_dates, buy_price, sell_dates, sell_price, positions

    def plot_moving_average(self, df, src, out1, buy_dates, buy_price, sell_dates, sell_price, atype_name):
        """Plotting the stock price and trading signals

        Args:
            src (dataframe): stock price for a given stock
            out1 (dataframe):  a specific MA for a given stock
            buy_dates (series): buy dates for a give stock under s specific MA
            buy_price (series): buy prices for a give stock under s specific MA
            sell_dates (series): sell dates for a give stock under s specific MA
            sell_price (series): sell prices for a give stock under s specific MA
            atype_name (str): the name of a given MA
        """
        
        plt.figure(figsize=(20, 8))
        plt.plot(df.index, src, label='Stock Price')
        plt.plot(df.index, out1, label='Moving Average 1')
        plt.plot(buy_dates, buy_price, '^', markersize=10, color='g', label='Buy Signal')
        plt.plot(sell_dates, sell_price, 'v', markersize=10, color='r', label='Sell Signal')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title('Stock Price with Trading Signals and Moving Average:' + str(atype_name))
        plt.legend()
        plt.grid(True)
        plt.show()
        
    def waveTrend(self, test_stock, n1, n2):
        """Compute the wave trend indicators

        Args:
            test_stock (dataframe): info for a given stock
            n1 (int): a period for a EWMA
            n2 (int): a period for a EWMA

        Returns:
            _type_: _description_
        """
        
        # Retrieve Apple stock data from 2018 to present
        data = test_stock

        # Extract the necessary columns

        high = data["high"].values
        low = data["low"].values
        close = data["close"].values
        dates = data.index

        # Calculate average price (hlc3)
        ap = (high + low + close) / 3

        # Calculate components
        esa = EMA(ap, n1)
        
        # represents the dominant cycle period and is calculated as the exponential 
        # moving average of the absolute difference between the average price and esa with a length of n1.
        d = EMA(np.abs(ap - esa), n1)
        
        # the cycle indicator, calculated as the difference between the average price and esa,
        # divided by 0.015 multiplied by d.
        ci = (ap - esa) / (0.015 * d)
        
        # represents the trend-cycle indicator,
        # which is the exponential moving average of ci with a length of n2.
        tci = EMA(ci, n2)

        wt1 = tci
        wt2 = EMA(wt1, 4)
        return wt1, wt2, dates, ap

    def buy_sell_signals_WT(self, Signals, wt1, wt2):
        """Build buy and sell signals for wave trend oscillator

        Args:
            Signals (dataframe): a dataframe to store signals info
            wt1 (dataframe): a EWMA
            wt2 (dataframe): a EWMA

        Returns:
            buy_dates, buy_positions, sell_dates, sell_positions
        """
        
        Signals['WT1'] = wt1
        Signals['WT2'] = wt2

        Signals['WaveTrend Buy']= np.where((Signals['WT1'].shift(1) < Signals['WT2'].shift(1)) & 
                                    (Signals['WT1'] > Signals['WT2']), 1, 0)
        buy_dates = (Signals.loc[Signals['WT1'] < -60].index.intersection(Signals.loc[Signals['WaveTrend Buy'] == 1].index))
        buy_positions = Signals.loc[buy_dates]['WT1']

        Signals['WaveTrend Sell']= np.where((Signals['WT1'].shift(1) > Signals['WT2'].shift(1)) &
                                    (Signals['WT1'] < Signals['WT2']), -1, 0)
        sell_dates = (Signals.loc[Signals['WT1'] > 60].index.intersection(Signals.loc[Signals['WaveTrend Sell'] == -1].index))
        sell_positions = Signals.loc[sell_dates]['WT1']
        
        return buy_dates, buy_positions, sell_dates, sell_positions

    def plot_wave_trend(self, dates, ap, obLevel1, obLevel2, osLevel1, osLevel2, buy_dates, buy_positions, sell_dates, sell_positions, wt1, wt2):
        """Plot wave trend oscillator

        Args:
            dates (series):
            ap (series):  the average price, calculated as the average of high, low, and close
            obLevel1 (int): Over Bought Level 1
            obLevel2 (int): Over Bought Level 2
            osLevel1 (int): Over Sold Level 1
            osLevel2 (int): Over Sold Level 2
            buy_dates (series): _description_
            buy_positions (series): _description_
            sell_dates (series): _description_
            sell_positions (series): _description_
            wt1 (dataframe):  the Wave Trend Oscillator's first component,
            wt2 (dataframe):  the second component of the oscillator,
                        calculated as the simple moving average of wt1 with a length of 4.
        """
        
        plt.figure(figsize=(20,8))
        plt.plot(dates, np.zeros(len(ap)), color='gray')
        plt.axhline(obLevel1, color='red')
        plt.axhline(osLevel1, color='green')
        plt.axhline(obLevel2, color='red', linestyle='dotted')
        plt.axhline(osLevel2, color='green', linestyle='dotted')

        plt.plot(buy_dates, buy_positions, '^', markersize=10, color='g', label='Buy Signal')
        plt.plot(sell_dates, sell_positions, 'v', markersize=10, color='r', label='Sell Signal')
        plt.plot(dates, wt1, color='green')
        plt.plot(dates, wt2, color='red', linestyle='dotted')

        plt.title("WaveTrend [LazyBear] for Apple Stock")
        plt.xlabel("Time")
        plt.ylabel("Oscillator Value")
        plt.legend(["Zero Line", "Overbought Level 1", "Oversold Level 1", "Overbought Level 2", "Oversold Level 2", "WT1", "WT2"])
        plt.show()

    def buy_sell_signals_MACD_RSI(self, test_stock, Signals):
        """Build buy and sell sginals for MACD and RSI

        Args:
            test_stock (dataframe): all info for a given stock
            Signals (dataframe): signals info for a given stock
        """
        
        rsi = RSI(test_stock['close'], timeperiod=14) 
        Signals['RSI'] = rsi
        Signals['RSI Buy'] = np.where(Signals['RSI'] < 30, 1, 0)
        Signals['RSI Sell'] = np.where(Signals['RSI'] > 70, -1, 0)
        macd, macdsignal, macdhist = MACD(test_stock['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        Signals['MACD'] = macd
        Signals['MACD_signal'] = macdsignal
        Signals['MACD Buy'] = np.where((Signals['MACD'].shift(1) < Signals['MACD_signal']) &
                                            (Signals['MACD'] > Signals['MACD_signal']), 1, 0)
        Signals['MACD Sell'] = np.where((Signals['MACD'].shift(1) > Signals['MACD_signal']) &
                                            (Signals['MACD'] < Signals['MACD_signal']), -1, 0)
        Signals.columns
        Signals.drop(['Signal',
                      'WT1',
                      'WT2',
                      'RSI',
                      'MACD',
                      'MACD_signal'], axis=1, inplace=True)
        Signals['Indicators'] = (Signals.sum(axis=1)/11)
        Signals.loc[Signals['Indicators'].between(-1,-0.5), 'Grade'] = 'Strong Sell'
        Signals.loc[Signals['Indicators'].between(-0.5,-0.1), 'Grade'] = 'Sell'
        Signals.loc[Signals['Indicators'].between(-0.1,0.1), 'Grade'] = 'Neutral'
        Signals.loc[Signals['Indicators'].between(0.1,0.5), 'Grade'] = 'Buy'
        Signals.loc[Signals['Indicators'].between(0.5,1), 'Grade'] = 'Strong Buy'

    def support_resistence(self, test_stock, Signals):
        """Compute support and resistence levels for a given stock

        Args:
            test_stock (dataframe): all info for a given stock
            Signals (dataframe): signals info for a given stock
            
        Returns:
            values for support and resistence levels
        """
        
        high_daily = test_stock['high'].values
        low_daily = test_stock['low'].values
        close_daily = test_stock['close'].values
        pivot_daily = (high_daily + low_daily + close_daily)/3
        R2 = pivot_daily + (high_daily - low_daily)
        S2 = pivot_daily - (high_daily - low_daily)
        fibonacci_R2 = pivot_daily + 0.68 * (high_daily - low_daily)
        fibonacci_S2 = pivot_daily - 0.68 * (high_daily - low_daily)
        pivot_frame = {'high_daily': high_daily,
                    'low_daily': low_daily,
                    'close_daily': close_daily,
                    'Pivot Classic': pivot_daily,
                    'R2': R2,
                    'S2': S2,
                    'Fibonacci R2': fibonacci_R2,
                    'Fibonacci S2': fibonacci_S2}
        Signals['Classic R2'] = R2
        Signals['Classic S2'] = S2
        Signals['Fibonacci R2'] = fibonacci_R2
        Signals['Fibonacci S2'] = fibonacci_S2
        Signals['Last Close'] = test_stock['close']
        return R2, S2, fibonacci_R2, fibonacci_S2, pivot_frame

    def plot_support_resistence(self, test_stock, classic_R2, classic_S2, fibonacci_R2, fibonacci_S2):
        """Plot support and resistence levels for a stock

        Args:
            test_stock (dataframe): all info for a given stock
            classic_R2 (series): classic pivot R2
            classic_S2 (series): classic pivot S2
            fibonacci_R2 (series): fibonacci R2
            fibonacci_S2 (series): fibonacci S2
        """
        
        plt.figure(figsize=(15,8))
        plt.plot(test_stock.index, fibonacci_R2)
        plt.plot(test_stock.index, fibonacci_S2)
        plt.plot(test_stock.index, classic_R2)
        plt.plot(test_stock.index, classic_S2)
        
    def notification_single_stock(self, Signals):
        """Build notification for a single stock

        Args:
            Signals (_type_): signals info for a given stock

        Returns:
            a single stock notification for all days (single_stock_notification),
            a single stock notification for the last day except for the 'neutral' state (final_signals)
        """
        
        columns = ['Last Close',
                   'Grade',
                   'Classic R2',
                   'Classic S2',
                   'Fibonacci R2',
                   'Fibonacci S2']
        final_signals = Signals.loc[:, columns]
        single_stock_notification = final_signals.loc[final_signals['Grade'] != 'Neutral'][-1:]
        return single_stock_notification, final_signals



# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from finta import TA
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from talib import RSI, MACD, BBANDS, WMA, EMA, T3, SMA
from datetime import date, datetime



# Initial Values
factorT3 = 7
length = 10
atype = 5
len_1 = 20
len_2 = 50
len1 = 20
factorT3 = 0.7
len2 = 50
sfactorT3 = 7
atype2 = 8
smoothe = 2
n1 = 10
n2 = 21
obLevel1 = 60
obLevel2 = 53
osLevel1 = -60
osLevel2 = -53
atypes = {1: 'Simple Moving Average',
          2: 'Exponential Moving Average (Not Adjusted)',
          3: 'Weighted Moving Average',
          4: 'Hull Moving Average',
          5: 'Volume Weighted Moving Average',
          6: 'Exponential Moving Average (Adjusted)',
          7: 'Triple Exponential Moving Average',
          8: 'Tilson T3 Moving Average'}
# Choose 'Daily' or 'Weekly' mode
dailyOrWeekly = "Daily" 
spc = False
cc = True
doma2 = False
spc2 = False
cc2 = True
warn = False
warn2 = False
sd = False
# Choose whether to plot
plot_RSI_BB_MACD = False
plot_moving_average = False
plot_wave_trend = False
plot_support_resistence = False




c = Indicators()

# Get ticker_names and stock info
tickers, sp500_daily = c.create_dataframe()

# Build a framework for notification system
data = [{'date':'test', 'Grade':'test', 'Classic R2':'test', 'Classic S2':'test', 'Fibonacci R2':'test', 'Fibonacci S2':'test'}]
test_stock_notification = pd.DataFrame(data)

tickers = tickers[:]
# Compute signals for technical indicators, support and resistence levels for all stocks
for ticker in tickers:
    test_stock, length = c.create_test_stock(ticker, sp500_daily, dailyOrWeekly)
    if length < 40: 
        continue
    df = test_stock
    test_price = test_stock.close
    src = test_price
    rsi, macd, macdsignal, macdhist, up, mid, low, data = c.RSI_BB_MACD(test_price)
    
    # Choose whether to plot
    if plot_RSI_BB_MACD:
        plot_RSI_BB_MACD(test_price, rsi, macd, macdsignal, macdhist, up, mid, low, data)

    # Build a dataframe to store signals, support and resistence levels
    Signals = pd.DataFrame(index=df.index)
    Signals['Signal'] = 0
    
    # Compute 8 moving averages and buy/sell dates and prices
    for atype, atype_name in atypes.items(): 
        src, out1= c.compute_moving_average(src, df, len1, atype, factorT3)
        buy_dates, buy_price, sell_dates, sell_price, positions = c.buy_sell_signals_MA(Signals, src, out1, atype_name)
        
        # Choose whether to plot
        if plot_moving_average:
            plot_moving_average(df, src, out1, buy_dates, buy_price, sell_dates, sell_price, atype_name)
        
    # Compute the wave trend oscillator and buy/sell dates and prices
    wt1, wt2, dates, ap = c.waveTrend(test_stock, n1, n2)
    buy_dates, buy_positions, sell_dates, sell_positions = c.buy_sell_signals_WT(Signals, wt1, wt2)
    
    # Choose whether to plot
    if plot_wave_trend:
        plot_wave_trend(dates, ap, obLevel1, obLevel2, osLevel1, osLevel2, buy_dates, buy_positions, sell_dates, sell_positions, wt1, wt2)
    c.buy_sell_signals_MACD_RSI(test_stock, Signals)
    classic_R2, classic_S2, fibonacci_R2, fibonacci_S2, pivot_frame = c.support_resistence(test_stock, Signals)
    Signals.dropna(how='any',inplace=True)
    
    # Choose whether to plot
    if plot_support_resistence: plot_support_resistence(test_stock, classic_R2, classic_S2, fibonacci_R2, fibonacci_S2)
    single_stock_notification, final_signals = c.notification_single_stock(Signals)
    
    # Build the notification system for all stocks
    single_stock_notification.reset_index(inplace=True)
    combined = pd.concat([test_stock_notification, single_stock_notification], ignore_index=True)
    test_stock_notification = combined

combined.drop(0, inplace=True)
combined_stocks_notification = combined.reset_index(drop=True)
tickers_df = pd.DataFrame(tickers, columns=['Ticker'])
combined_stocks_notification_csv = pd.concat([combined_stocks_notification, tickers_df], axis=1)
combined_stocks_notification_csv['days'] = (datetime.today() - pd.to_datetime(combined_stocks_notification_csv.date)).dt.days
order = ['Ticker', 'Last Close', 'Grade', 'Classic R2', 'Classic S2', 'Fibonacci R2', 'Fibonacci S2', 'days', 'date']
combined_stocks_notification_csv = combined_stocks_notification_csv[order]
combined_stocks_notification_csv.to_excel('[' + dailyOrWeekly + '] ' + 'combined_stocks_notification_csv.xlsx')
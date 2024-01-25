import pandas as pd
import datetime
import numpy as np
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
from get_data import *

threshold_volumn = 459.065
threshold_sell = 245.513
threshold_buy = 246.183
threshold_volumn_r = 14377198.20746
threshold_buy_r = 7744991.921
threshold_sell_r = 7767176.11029
threshold_volumn_count = 2954.0
threshold_buy_count = 1528.0
threshold_sell_count = 1544.0

def get_indicators(df):

    df['volumn'] = df['sell'] + df['buy']
    df['volumn_r'] = df['sell_r'] + df['buy_r']
    df['volumn_count'] = df['sell_count'] + df['buy_count']

    # 可能的目标变量
    # 前1%volumn样本
    df['volumn_top_1pct'] = 0
    df.loc[df['volumn'] >= threshold_volumn, 'volumn_top_1pct'] = 1

    # 前1%sell样本
    df['sell_top_1pct'] = 0
    df.loc[df['sell'] >= threshold_sell, 'sell_top_1pct'] = 1

    # 前1%buy样本
    df['buy_top_1pct'] = 0
    df.loc[df['buy'] >= threshold_buy, 'buy_top_1pct'] = 1

    # 前1%volumn_r样本
    df['volumn_r_top_1pct'] = 0
    df.loc[df['volumn_r'] >= threshold_volumn_r, 'volumn_r_top_1pct'] = 1

    # 前1%buy_r样本
    df['buy_r_top_1pct'] = 0
    df.loc[df['buy_r'] >= threshold_buy_r, 'buy_r_top_1pct'] = 1

    # 前1%sell_r样本
    df['sell_r_top_1pct'] = 0
    df.loc[df['sell_r'] >= threshold_sell_r, 'sell_r_top_1pct'] = 1

    # 前1%volumn_count样本
    df['volumn_count_top_1pct'] = 0
    df.loc[df['volumn_count'] >= threshold_volumn_count, 'volumn_count_top_1pct'] = 1

    # 前1%buy_count样本
    df['buy_count_top_1pct'] = 0
    df.loc[df['buy_count'] >= threshold_buy_count, 'buy_count_top_1pct'] = 1

    # 前1%sell_count样本
    df['sell_count_top_1pct'] = 0
    df.loc[df['sell_count'] >= threshold_sell_count, 'sell_count_top_1pct'] = 1

    # 全部前1%
    df['all_1pct'] = 0
    df.loc[(df['volumn_top_1pct']==1)&(df['sell_top_1pct']==1)&(df['buy_top_1pct']==1)&(df['volumn_r_top_1pct']==1)&(df['buy_r_top_1pct']==1)&(df['sell_r_top_1pct']==1)&(df['volumn_count_top_1pct']==1)&(df['buy_count_top_1pct']==1)&(df['sell_count_top_1pct']==1),'all_1pct'] = 1


    # volumn相关前1%
    df['volumn_all_1pct'] = 0
    df.loc[(df['volumn_top_1pct']==1)&(df['volumn_r_top_1pct']==1)&(df['volumn_count_top_1pct']==1),'volumn_all_1pct'] = 1
    # buy相关前1%
    df['buy_all_1pct'] = 0
    df.loc[(df['buy_top_1pct']==1)&(df['buy_r_top_1pct']==1)&(df['buy_count_top_1pct']==1),'buy_all_1pct'] = 1
    # sell相关前1%
    df['sell_all_1pct'] = 0
    df.loc[(df['sell_top_1pct']==1)&(df['sell_r_top_1pct']==1)&(df['sell_count_top_1pct']==1),'sell_all_1pct'] = 1



    # Indicators
    # 计算前5分钟'sell'值的和
    rolling_sell_sum = df['sell'].rolling('5T').sum()
    # sell5min变动
    df['sell_5m_5m_1'] = (rolling_sell_sum/rolling_sell_sum.shift(30)-1)
    df['sell_5m_1m_1'] = (rolling_sell_sum/rolling_sell_sum.shift(6)-1)
    df['sell_5m_10s_1'] = (rolling_sell_sum/rolling_sell_sum.shift(1)-1)

    # sell5min变动----shift 10
    df['sell_5m_5m_10'] = (rolling_sell_sum/rolling_sell_sum.shift(30)-1).shift(10)
    df['sell_5m_1m_10'] = (rolling_sell_sum/rolling_sell_sum.shift(6)-1).shift(10)
    df['sell_5m_10s_10'] = (rolling_sell_sum/rolling_sell_sum.shift(1)-1).shift(10)

    # 计算前1分钟'sell'值的和
    rolling_sell_sum_1 = df['sell'].rolling('1T').sum()
    # sell1min变动
    df['sell_1m_5m_1'] = (rolling_sell_sum_1/rolling_sell_sum_1.shift(30)-1)
    df['sell_1m_1m_1'] = (rolling_sell_sum_1/rolling_sell_sum_1.shift(6)-1)
    df['sell_1m_10s_1'] = (rolling_sell_sum_1/rolling_sell_sum_1.shift(1)-1)

    # sell1min变动----shift 10
    df['sell_1m_5m_10'] = (rolling_sell_sum_1/rolling_sell_sum_1.shift(30)-1).shift(10)
    df['sell_1m_1m_10'] = (rolling_sell_sum_1/rolling_sell_sum_1.shift(6)-1).shift(10)
    df['sell_1m_10s_10'] = (rolling_sell_sum_1/rolling_sell_sum_1.shift(1)-1).shift(10)

    # 计算sell变动
    df['sell_change_10s_1'] = df['sell'].diff()
    df['sell_change_60s_1'] = (df['sell']-df['sell'].shift(6))
    df['sell_change_300s_1'] = (df['sell']-df['sell'].shift(30))

    # 计算sell变动----shift 10
    df['sell_change_10s_10'] = df['sell'].diff().shift(10)
    df['sell_change_60s_10'] = (df['sell']-df['sell'].shift(6)).shift(10)
    df['sell_change_300s_10'] = (df['sell']-df['sell'].shift(30)).shift(10)

    # 计算sell变动率
    df['sell_change_re_10s_1'] = (df['sell'].diff()/df['sell'].shift(1))
    df['sell_change_re_60s_1'] = ((df['sell']-df['sell'].shift(6))/df['sell'].shift(6))
    df['sell_change_re_300s_1'] = ((df['sell']-df['sell'].shift(30))/df['sell'].shift(30))

    # 计算sell变动率----shift 10
    df['sell_change_re_10s_10'] = (df['sell'].diff()/df['sell'].shift(1)).shift(10)
    df['sell_change_re_60s_10'] = ((df['sell']-df['sell'].shift(6))/df['sell'].shift(6)).shift(10)
    df['sell_change_re_300s_10'] = ((df['sell']-df['sell'].shift(30))/df['sell'].shift(30)).shift(10)

    # 计算sell波动率
    df['sell_volatility_50s_1'] = df['sell'].diff().rolling(window=5).std()
    df['sell_volatility_200s_1'] = df['sell'].diff().rolling(window=20).std()
    df['sell_volatility_600s_1'] = df['sell'].diff().rolling(window=60).std()
    # 计算sell波动率----shift 10
    df['sell_volatility_50s_10'] = df['sell'].diff().rolling(window=5).std().shift(10)
    df['sell_volatility_200s_10'] = df['sell'].diff().rolling(window=20).std().shift(10)
    df['sell_volatility_600s_10'] = df['sell'].diff().rolling(window=60).std().shift(10)

    # 计算前5分钟'buy'值的和
    rolling_buy_sum = df['buy'].rolling('5T').sum()
    # buy5min变动
    df['buy_5m_5m_1'] = (rolling_buy_sum/rolling_buy_sum.shift(30)-1)
    df['buy_5m_1m_1'] = (rolling_buy_sum/rolling_buy_sum.shift(6)-1)
    df['buy_5m_10s_1'] = (rolling_buy_sum/rolling_buy_sum.shift(1)-1)

    # buy5min变动----shift 10
    df['buy_5m_5m_10'] = (rolling_buy_sum/rolling_buy_sum.shift(30)-1).shift(10)
    df['buy_5m_1m_10'] = (rolling_buy_sum/rolling_buy_sum.shift(6)-1).shift(10)
    df['buy_5m_10s_10'] = (rolling_buy_sum/rolling_buy_sum.shift(1)-1).shift(10)

    # 计算前1分钟'sell'值的和
    rolling_buy_sum_1 = df['buy'].rolling('1T').sum()
    # buy1min变动
    df['buy_1m_5m_1'] = (rolling_buy_sum_1/rolling_buy_sum_1.shift(30)-1)
    df['buy_1m_1m_1'] = (rolling_buy_sum_1/rolling_buy_sum_1.shift(6)-1)
    df['buy_1m_10s_1'] = (rolling_buy_sum_1/rolling_buy_sum_1.shift(1)-1)

    # buy1min变动----shift 10
    df['buy_1m_5m_10'] = (rolling_buy_sum_1/rolling_buy_sum_1.shift(30)-1).shift(10)
    df['buy_1m_1m_10'] = (rolling_buy_sum_1/rolling_buy_sum_1.shift(6)-1).shift(10)
    df['buy_1m_10s_10'] = (rolling_buy_sum_1/rolling_buy_sum_1.shift(1)-1).shift(10)


    # 计算buy变动
    df['buy_change_10s_1'] = df['buy'].diff()
    df['buy_change_60s_1'] = (df['buy']-df['buy'].shift(6))
    df['buy_change_300s_1'] = (df['buy']-df['buy'].shift(30))

    # 计算buy变动----shift 10
    df['buy_change_10s_10'] = df['buy'].diff().shift(10)
    df['buy_change_60s_10'] = (df['buy']-df['buy'].shift(6)).shift(10)
    df['buy_change_300s_10'] = (df['buy']-df['buy'].shift(30)).shift(10)

    # 计算buy变动率
    df['buy_change_re_10s_1'] = (df['buy'].diff()/df['buy'].shift(1))
    df['buy_change_re_60s_1'] = ((df['buy']-df['buy'].shift(6))/df['buy'].shift(6))
    df['buy_change_re_300s_1'] = ((df['buy']-df['buy'].shift(30))/df['buy'].shift(30))

    # 计算buy变动率----shift 10
    df['buy_change_re_10s_10'] = (df['buy'].diff()/df['buy'].shift(1)).shift(10)
    df['buy_change_re_60s_10'] = ((df['buy']-df['buy'].shift(6))/df['buy'].shift(6)).shift(10)
    df['buy_change_re_300s_10'] = ((df['buy']-df['buy'].shift(30))/df['buy'].shift(30)).shift(10)

    # 计算buy波动率
    df['buy_volatility_50s_1'] = df['buy'].diff().rolling(window=5).std()
    df['buy_volatility_200s_1'] = df['buy'].diff().rolling(window=20).std()
    df['buy_volatility_600s_1'] = df['buy'].diff().rolling(window=60).std()
    # 计算buy波动率----shift 10
    df['buy_volatility_50s_10'] = df['buy'].diff().rolling(window=5).std().shift(10)
    df['buy_volatility_200s_10'] = df['buy'].diff().rolling(window=20).std().shift(10)
    df['buy_volatility_600s_10'] = df['buy'].diff().rolling(window=60).std().shift(10)

    # 计算1分钟buy和的波动率
    df['buy_volatility_1m_sum_100s_1'] = rolling_buy_sum_1.diff().rolling(window=10).std()
    df['buy_volatility_1m_sum_300s_1'] = rolling_buy_sum_1.diff().rolling(window=30).std()
    df['buy_volatility_1m_sum_1000s_1'] = rolling_buy_sum_1.diff().rolling(window=100).std()
    # 计算1分钟buy和的波动率----shift 10
    df['buy_volatility_1m_sum_100s_10'] = rolling_buy_sum_1.diff().rolling(window=10).std().shift(10)
    df['buy_volatility_1m_sum_300s_10'] = rolling_buy_sum_1.diff().rolling(window=30).std().shift(10)
    df['buy_volatility_1m_sum_1000s_10'] = rolling_buy_sum_1.diff().rolling(window=100).std().shift(10)

    # 计算5分钟buy和的波动率
    df['buy_volatility_5m_sum_100s_1'] = rolling_buy_sum.diff().rolling(window=10).std()
    df['buy_volatility_5m_sum_300s_1'] = rolling_buy_sum.diff().rolling(window=30).std()
    # 计算5分钟buy和的波动率----shift 10
    df['buy_volatility_5m_sum_100s_10'] = rolling_buy_sum.diff().rolling(window=10).std().shift(10)
    df['buy_volatility_5m_sum_300s_10'] = rolling_buy_sum.diff().rolling(window=30).std().shift(10)


    # 计算买卖之差占比（市场情绪）
    df['market_sentiment_1'] = ((df['buy'] - df['sell']) / (df['buy'] + df['sell']))
    # 计算买卖之差占比（市场情绪）----shift 10
    df['market_sentiment_10'] = ((df['buy'] - df['sell']) / (df['buy'] + df['sell'])).shift(10)
    # 计算买卖之差占比（市场情绪）----shift 5
    df['market_sentiment_5'] = ((df['buy'] - df['sell']) / (df['buy'] + df['sell'])).shift(5)
    # 计算买卖之差占比（市场情绪）----shift 60
    df['market_sentimen0t_60'] = ((df['buy'] - df['sell']) / (df['buy'] + df['sell'])).shift(60)
    # 计算买卖之差占比（市场情绪）----shift 120
    df['market_sentiment_120'] = ((df['buy'] - df['sell']) / (df['buy'] + df['sell'])).shift(120)

    # volumn 
    volumn = df['sell'] + df['buy']
    volumn_count = df['sell_count'] + df['buy_count']

    # 计算前5分钟'volumn'值的和
    rolling_volumn_sum = volumn.rolling('5T').sum()
    # volumn5min变动
    df['volumn_5m_5m_1'] = (rolling_volumn_sum/rolling_volumn_sum.shift(30)-1)
    df['volumn_5m_1m_1'] = (rolling_volumn_sum/rolling_volumn_sum.shift(6)-1)
    df['volumn_5m_10s_1'] = (rolling_volumn_sum/rolling_volumn_sum.shift(1)-1)

    # volumn5min变动----shift 10
    df['volumn_5m_5m_10'] = (rolling_volumn_sum/rolling_volumn_sum.shift(30)-1).shift(10)
    df['volumn_5m_1m_10'] = (rolling_volumn_sum/rolling_volumn_sum.shift(6)-1).shift(10)
    df['volumn_5m_10s_10'] = (rolling_volumn_sum/rolling_volumn_sum.shift(1)-1).shift(10)

    # 计算前1分钟'volumn'值的和
    rolling_volumn_sum_1 = volumn.rolling('1T').sum()
    # volumn1min变动
    df['volumn_1m_5m_1'] = (rolling_volumn_sum_1/rolling_volumn_sum_1.shift(30)-1)
    df['volumn_1m_1m_1'] = (rolling_volumn_sum_1/rolling_volumn_sum_1.shift(6)-1)
    df['volumn_1m_10s_1'] = (rolling_volumn_sum_1/rolling_volumn_sum_1.shift(1)-1)

    # volumn1min变动----shift 10
    df['volumn_1m_5m_10'] = (rolling_volumn_sum_1/rolling_volumn_sum_1.shift(30)-1).shift(10)
    df['volumn_1m_1m_10'] = (rolling_volumn_sum_1/rolling_volumn_sum_1.shift(6)-1).shift(10)
    df['volumn_1m_10s_10'] = (rolling_volumn_sum_1/rolling_volumn_sum_1.shift(1)-1).shift(10)


    # 计算volumn变动
    df['volumn_change_10s_1'] = volumn.diff()
    df['volumn_change_60s_1'] = (volumn-volumn.shift(6))
    df['volumn_change_300s_1'] = (volumn-volumn.shift(30))

    # 计算volumn变动----shift 10
    df['volumn_change_10s_10'] = volumn.diff().shift(10)
    df['volumn_change_60s_10'] = (volumn-volumn.shift(6)).shift(10)
    df['volumn_change_300s_10'] = (volumn-volumn.shift(30)).shift(10)

    # 计算volumn变动率
    df['volumn_change_re_10s_1'] = (volumn.diff()/volumn.shift(1))
    df['volumn_change_re_60s_1'] = ((volumn-volumn.shift(6))/volumn.shift(6))
    df['volumn_change_re_300s_1'] = ((volumn-volumn.shift(30))/volumn.shift(30))

    # 计算volumn变动率----shift 10
    df['volumn_change_re_10s_10'] = (volumn.diff()/volumn.shift(1)).shift(10)
    df['volumn_change_re_60s_10'] = ((volumn-volumn.shift(6))/volumn.shift(6)).shift(10)
    df['volumn_change_re_300s_10'] = ((volumn-volumn.shift(30))/volumn.shift(30)).shift(10)

    # 计算volumn波动率
    df['volumn_volatility_50s_1'] = volumn.diff().rolling(window=5).std()
    df['volumn_volatility_200s_1'] = volumn.diff().rolling(window=20).std()
    df['volumn_volatility_600s_1'] = volumn.diff().rolling(window=60).std()
    # 计算volumn波动率----shift 10
    df['volumn_volatility_50s_10'] = volumn.diff().rolling(window=5).std().shift(10)
    df['volumn_volatility_200s_10'] = volumn.diff().rolling(window=20).std().shift(10)
    df['volumn_volatility_600s_10'] = volumn.diff().rolling(window=60).std().shift(10)

    # 计算1分钟volumn和的波动率
    df['volumn_volatility_1m_sum_100s_1'] = rolling_volumn_sum_1.diff().rolling(window=10).std()
    df['volumn_volatility_1m_sum_300s_1'] = rolling_volumn_sum_1.diff().rolling(window=30).std()
    df['volumn_volatility_1m_sum_1000s_1'] = rolling_volumn_sum_1.diff().rolling(window=100).std()
    # 计算1分钟volumn和的波动率----shift 10
    df['volumn_volatility_1m_sum_100s_10'] = rolling_volumn_sum_1.diff().rolling(window=10).std().shift(10)
    df['volumn_volatility_1m_sum_300s_10'] = rolling_volumn_sum_1.diff().rolling(window=30).std().shift(10)
    df['volumn_volatility_1m_sum_1000s_10'] = rolling_volumn_sum_1.diff().rolling(window=100).std().shift(10)

    # 计算5分钟volumn和的波动率
    df['volumn_volatility_5m_sum_100s_1'] = rolling_volumn_sum.diff().rolling(window=10).std()
    df['volumn_volatility_5m_sum_300s_1'] = rolling_volumn_sum.diff().rolling(window=30).std()
    df['volumn_volatility_5m_sum_1000s_1'] = rolling_volumn_sum.diff().rolling(window=100).std()
    # 计算5分钟volumn和的波动率----shift 10
    df['volumn_volatility_5m_sum_100s_10'] = rolling_volumn_sum.diff().rolling(window=10).std().shift(10)
    df['volumn_volatility_5m_sum_300s_10'] = rolling_volumn_sum.diff().rolling(window=30).std().shift(10)
    df['volumn_volatility_5m_sum_1000s_10'] = rolling_volumn_sum.diff().rolling(window=100).std().shift(10)


    # 计算前5分钟'volumn_count'值的和
    rolling_volumn_count_sum = volumn_count.rolling('5T').sum()
    # volumn_count5min变动
    df['volumn_count_5m_5m_1'] = (rolling_volumn_count_sum/rolling_volumn_count_sum.shift(30)-1)
    df['volumn_count_5m_1m_1'] = (rolling_volumn_count_sum/rolling_volumn_count_sum.shift(6)-1)
    df['volumn_count_5m_10s_1'] = (rolling_volumn_count_sum/rolling_volumn_count_sum.shift(1)-1)

    # volumn_count5min变动----shift 10
    df['volumn_count_5m_5m_10'] = (rolling_volumn_count_sum/rolling_volumn_count_sum.shift(30)-1).shift(10)
    df['volumn_count_5m_1m_10'] = (rolling_volumn_count_sum/rolling_volumn_count_sum.shift(6)-1).shift(10)
    df['volumn_count_5m_10s_10'] = (rolling_volumn_count_sum/rolling_volumn_count_sum.shift(1)-1).shift(10)

    # 计算前1分钟'volumn_count'值的和
    rolling_volumn_count_sum_1 = volumn_count.rolling('1T').sum()
    # volumn_count1min变动
    df['volumn_count_1m_5m_1'] = (rolling_volumn_count_sum_1/rolling_volumn_count_sum_1.shift(30)-1)
    df['volumn_count_1m_1m_1'] = (rolling_volumn_count_sum_1/rolling_volumn_count_sum_1.shift(6)-1)
    df['volumn_count_1m_10s_1'] = (rolling_volumn_count_sum_1/rolling_volumn_count_sum_1.shift(1)-1)

    # volumn_count1min变动----shift 10
    df['volumn_count_1m_5m_10'] = (rolling_volumn_count_sum_1/rolling_volumn_count_sum_1.shift(30)-1).shift(10)
    df['volumn_count_1m_1m_10'] = (rolling_volumn_count_sum_1/rolling_volumn_count_sum_1.shift(6)-1).shift(10)
    df['volumn_count_1m_10s_10'] = (rolling_volumn_count_sum_1/rolling_volumn_count_sum_1.shift(1)-1).shift(10)


    # 计算volumn_count变动
    df['volumn_count_change_10s_1'] = volumn_count.diff()
    df['volumn_count_change_60s_1'] = (volumn_count-volumn_count.shift(6))
    df['volumn_count_change_300s_1'] = (volumn_count-volumn_count.shift(30))

    # 计算volumn_count变动----shift 10
    df['volumn_count_change_10s_10'] = volumn_count.diff().shift(10)
    df['volumn_count_change_60s_10'] = (volumn_count-volumn_count.shift(6)).shift(10)
    df['volumn_count_change_300s_10'] = (volumn_count-volumn_count.shift(30)).shift(10)

    # 计算volumn_count变动率
    df['volumn_count_change_re_10s_1'] = (volumn_count.diff()/volumn_count.shift(1))
    df['volumn_count_change_re_60s_1'] = ((volumn_count-volumn_count.shift(6))/volumn_count.shift(6))
    df['volumn_count_change_re_300s_1'] = ((volumn_count-volumn_count.shift(30))/volumn_count.shift(30))

    # 计算volumn_count变动率----shift 10
    df['volumn_count_change_re_10s_10'] = (volumn_count.diff()/volumn_count.shift(1)).shift(10)
    df['volumn_count_change_re_60s_10'] = ((volumn_count-volumn_count.shift(6))/volumn_count.shift(6)).shift(10)
    df['volumn_count_change_re_300s_10'] = ((volumn_count-volumn_count.shift(30))/volumn_count.shift(30)).shift(10)

    # 计算volumn_count波动率
    df['volumn_count_volatility_50s_1'] = volumn_count.diff().rolling(window=5).std()
    df['volumn_count_volatility_200s_1'] = volumn_count.diff().rolling(window=20).std()
    df['volumn_count_volatility_600s_1'] = volumn_count.diff().rolling(window=60).std()
    # 计算volumn_count波动率----shift 10
    df['volumn_count_volatility_50s_10'] = volumn_count.diff().rolling(window=5).std().shift(10)
    df['volumn_count_volatility_200s_10'] = volumn_count.diff().rolling(window=20).std().shift(10)
    df['volumn_count_volatility_600s_10'] = volumn_count.diff().rolling(window=60).std().shift(10)

    # 计算1分钟volumn_count和的波动率
    df['volumn_count_volatility_1m_sum_100s_1'] = rolling_volumn_count_sum_1.diff().rolling(window=10).std()
    df['volumn_count_volatility_1m_sum_300s_1'] = rolling_volumn_count_sum_1.diff().rolling(window=30).std()
    df['volumn_count_volatility_1m_sum_1000s_1'] = rolling_volumn_count_sum_1.diff().rolling(window=100).std()
    # 计算1分钟volumn_count和的波动率----shift 10
    df['volumn_count_volatility_1m_sum_100s_10'] = rolling_volumn_count_sum_1.diff().rolling(window=10).std().shift(10)
    df['volumn_count_volatility_1m_sum_300s_10'] = rolling_volumn_count_sum_1.diff().rolling(window=30).std().shift(10)
    df['volumn_count_volatility_1m_sum_1000s_10'] = rolling_volumn_count_sum_1.diff().rolling(window=100).std().shift(10)

    # 计算5分钟volumn_count和的波动率
    df['volumn_count_volatility_5m_sum_100s_1'] = rolling_volumn_count_sum.diff().rolling(window=10).std()
    df['volumn_count_volatility_5m_sum_300s_1'] = rolling_volumn_count_sum.diff().rolling(window=30).std()
    df['volumn_count_volatility_5m_sum_1000s_1'] = rolling_volumn_count_sum.diff().rolling(window=100).std()
    # 计算5分钟volumn_count和的波动率----shift 10
    df['volumn_count_volatility_5m_sum_100s_10'] = rolling_volumn_count_sum.diff().rolling(window=10).std().shift(10)
    df['volumn_count_volatility_5m_sum_300s_10'] = rolling_volumn_count_sum.diff().rolling(window=30).std().shift(10)
    df['volumn_count_volatility_5m_sum_1000s_10'] = rolling_volumn_count_sum.diff().rolling(window=100).std().shift(10)


    
    # 计算价格变动
    price_change_10s = df['min']-df['max']
    price_change_60s = df['min']-df['max'].shift(6)
    price_change_300s = df['min']-df['max'].shift(30)

    # 价格涨幅
    df['price_rise'] = (price_change_10s/df['max'].shift(1)-1)
    # 计算min变动
    df['price_rise_10s_1'] = (df['min']-df['max'].shift(1))
    df['price_rise_60s_1'] = (df['min']-df['max'].shift(6))
    df['price_rise_300s_1'] = (df['min']-df['max'].shift(30))

    # 计算min变动----shift 10
    df['price_rise_10s_10'] = (df['min']-df['max'].shift(1)).shift(10)
    df['price_rise_60s_10'] = (df['min']-df['max'].shift(6)).shift(10)
    df['price_rise_300s_10'] = (df['min']-df['max'].shift(30)).shift(10)

    # 计算min变动率
    df['price_rise_re_10s_1'] = ((df['min']-df['max'].shift(1))/df['max'].shift(1))
    df['price_rise_re_60s_1'] = ((df['min']-df['max'].shift(6))/df['max'].shift(6))
    df['price_rise_re_300s_1'] = ((df['min']-df['max'].shift(30))/df['max'].shift(30))

    # 计算min变动率----shift 10
    df['price_rise_re_10s_10'] = ((df['min']-df['max'].shift(1))/df['max'].shift(1)).shift(10)
    df['price_rise_re_60s_10'] = ((df['min']-df['max'].shift(6))/df['max'].shift(6)).shift(10)
    df['price_rise_re_300s_10'] = ((df['min']-df['max'].shift(30))/df['max'].shift(30)).shift(10)

    # 计算10s高低价格波动率
    df['price_volatility_10_5s_1'] = price_change_10s.rolling(window=5).std()
    df['price_volatility_10_10s_1'] = price_change_10s.rolling(window=10).std()
    df['price_volatility_10_30s_1'] = price_change_10s.rolling(window=30).std()
    df['price_volatility_10_60s_1'] = price_change_10s.rolling(window=60).std()
    # 计算10s高低价格波动率----shift 10
    df['price_volatility_10_5s_10'] = price_change_10s.rolling(window=5).std().shift(10)
    df['price_volatility_10_10s_10'] = price_change_10s.rolling(window=10).std().shift(10)
    df['price_volatility_10_30s_10'] = price_change_10s.rolling(window=30).std().shift(10)
    df['price_volatility_10_60s_10'] = price_change_10s.rolling(window=60).std().shift(10)

    # 计算60s高低价格波动率-
    df['price_volatility_60_5s_1'] = price_change_60s.rolling(window=5).std()
    df['price_volatility_60_10s_1'] = price_change_60s.rolling(window=10).std()
    df['price_volatility_60_30s_1'] = price_change_60s.rolling(window=30).std()
    df['price_volatility_60_60s_1'] = price_change_60s.rolling(window=60).std()
    # 计算60s高低价格波动率----shift 10
    df['price_volatility_60_5s_10'] = price_change_60s.rolling(window=5).std().shift(10)
    df['price_volatility_60_10s_10'] = price_change_60s.rolling(window=10).std().shift(10)
    df['price_volatility_60_30s_10'] = price_change_60s.rolling(window=30).std().shift(10)
    df['price_volatility_60_60s_10'] = price_change_60s.rolling(window=60).std().shift(10)

    # 计算300s高低价格波动率
    df['price_volatility_300_5s_1'] = price_change_300s.rolling(window=5).std()
    df['price_volatility_300_10s_1'] = price_change_300s.rolling(window=10).std()
    df['price_volatility_300_30s_1'] = price_change_300s.rolling(window=30).std()
    df['price_volatility_300_60s_1'] = price_change_300s.rolling(window=60).std()
    # 计算60s高低价格波动率----shift 10
    df['price_volatility_300_5s_10'] = price_change_300s.rolling(window=5).std().shift(10)
    df['price_volatility_300_10s_10'] = price_change_300s.rolling(window=10).std().shift(10)
    df['price_volatility_300_30s_10'] = price_change_300s.rolling(window=30).std().shift(10)
    df['price_volatility_300_60s_10'] = price_change_300s.rolling(window=60).std().shift(10)

    # 计算min变动
    df['min_change_10s_1'] = df['min'].diff()
    df['min_change_60s_1'] = (df['min']-df['min'].shift(6))
    df['min_change_300s_1'] = (df['min']-df['min'].shift(30))

    # 计算min变动----shift 10
    df['min_change_10s_10'] = df['min'].diff().shift(10)
    df['min_change_60s_10'] = (df['min']-df['min'].shift(6)).shift(10)
    df['min_change_300s_10'] = (df['min']-df['min'].shift(30)).shift(10)

    # 计算min变动率
    df['min_change_re_10s_1'] = (df['min'].diff()/df['min'].shift(1))
    df['min_change_re_60s_1'] = ((df['min']-df['min'].shift(6))/df['min'].shift(6))
    df['min_change_re_300s_1'] = ((df['min']-df['min'].shift(30))/df['min'].shift(30))

    # 计算min变动率----shift 10
    df['min_change_re_10s_10'] = (df['min'].diff()/df['min'].shift(1)).shift(10)
    df['min_change_re_60s_10'] = ((df['min']-df['min'].shift(6))/df['min'].shift(6)).shift(10)
    df['min_change_re_300s_10'] = ((df['min']-df['min'].shift(30))/df['min'].shift(30)).shift(10)

    # 计算min波动率
    df['min_volatility_50s_1'] = df['min'].diff().rolling(window=5).std()
    df['min_volatility_200s_1'] = df['min'].diff().rolling(window=20).std()
    df['min_volatility_1000s_1'] = df['min'].diff().rolling(window=100).std()
    # 计算min波动率----shift 10
    df['min_volatility_50s_10'] = df['min'].diff().rolling(window=5).std().shift(10)
    df['min_volatility_200s_10'] = df['min'].diff().rolling(window=20).std().shift(10)
    df['min_volatility_1000s_10'] = df['min'].diff().rolling(window=100).std().shift(10)

    # 计算max变动
    df['max_change_10s_1'] = df['max'].diff()
    df['max_change_60s_1'] = (df['max']-df['max'].shift(6))
    df['max_change_300s_1'] = (df['max']-df['max'].shift(30))

    # 计算max变动----shift 10
    df['max_change_10s_10'] = df['max'].diff().shift(10)
    df['max_change_60s_10'] = (df['max']-df['max'].shift(6)).shift(10)
    df['max_change_300s_10'] = (df['max']-df['max'].shift(30)).shift(10)

    # 计算max变动率
    df['max_change_re_10s_1'] = (df['max'].diff()/df['max'].shift(1))
    df['max_change_re_60s_1'] = ((df['max']-df['max'].shift(6))/df['max'].shift(6))
    df['max_change_re_300s_1'] = ((df['max']-df['max'].shift(30))/df['max'].shift(30))

    # 计算max变动率----shift 10
    df['max_change_re_10s_10'] = (df['max'].diff()/df['max'].shift(1)).shift(10)
    df['max_change_re_60s_10'] = ((df['max']-df['max'].shift(6))/df['max'].shift(6)).shift(10)
    df['max_change_re_300s_10'] = ((df['max']-df['max'].shift(30))/df['max'].shift(30)).shift(10)

    # 计算max波动率
    df['max_volatility_50s_1'] = df['max'].diff().rolling(window=5).std()
    df['max_volatility_200s_1'] = df['max'].diff().rolling(window=20).std()
    df['max_volatility_1000s_1'] = df['max'].diff().rolling(window=100).std()
    # 计算max波动率----shift 10
    df['max_volatility_50s_10'] = df['max'].diff().rolling(window=5).std().shift(10)
    df['max_volatility_200s_10'] = df['max'].diff().rolling(window=20).std().shift(10)
    df['max_volatility_1000s_10'] = df['max'].diff().rolling(window=100).std().shift(10)

    # 价格区间
    price_range = df['min']-df['max']
    df['price_range_10'] = price_range
    df['price_range_60'] = price_range.shift(6)
    df['price_range_300'] = price_range.shift(30)

    # 计算price_range变动
    df['price_range_change_10s_1'] = price_range.diff()
    df['price_range_change_60s_1'] = (price_range-price_range.shift(6))
    df['price_range_change_300s_1'] = (price_range-price_range.shift(30))

    # 计算price_range变动----shift 10
    df['price_range_change_10s_10'] = price_range.diff().shift(10)
    df['price_range_change_60s_10'] = (price_range-price_range.shift(6)).shift(10)
    df['price_range_change_300s_10'] = (price_range-price_range.shift(30)).shift(10)

    # 计算price_range变动率
    df['price_range_change_re_10s_1'] = (price_range.diff()/price_range.shift(1))
    df['price_range_change_re_60s_1'] = ((price_range-price_range.shift(6))/price_range.shift(6))
    df['price_range_change_re_300s_1'] = ((price_range-price_range.shift(30))/price_range.shift(30))

    # 计算price_range变动率----shift 10
    df['price_range_change_re_10s_10'] = (price_range.diff()/price_range.shift(1)).shift(10)
    df['price_range_change_re_60s_10'] = ((price_range-price_range.shift(6))/price_range.shift(6)).shift(10)
    df['price_range_change_re_300s_10'] = ((price_range-price_range.shift(30))/price_range.shift(30)).shift(10)

    # 计算price_range波动率
    df['price_range_volatility_50s_1'] = price_range.diff().rolling(window=5).std()
    df['price_range_volatility_200s_1'] = price_range.diff().rolling(window=20).std()
    df['price_range_volatility_1000s_1'] = price_range.diff().rolling(window=100).std()
    # 计算price_range波动率----shift 10
    df['price_range_volatility_50s_10'] = price_range.diff().rolling(window=5).std().shift(10)
    df['price_range_volatility_200s_10'] = price_range.diff().rolling(window=20).std().shift(10)
    df['price_range_volatility_1000s_10'] = price_range.diff().rolling(window=100).std().shift(10)

    # close1分钟移动平均线
    df['close_ma_1m_1'] = df['close'].rolling(window=6).mean()
    df['close_ma_1m_5'] = df['close'].rolling(window=6).mean().shift(5)
    df['close_ma_1m_10'] = df['close'].rolling(window=6).mean().shift(10)
    # close5分钟移动平均线
    df['close_ma_5m_1'] = df['close'].rolling(window=30).mean()
    df['close_ma_5m_5'] = df['close'].rolling(window=30).mean().shift(5)
    df['close_ma_5m_10'] = df['close'].rolling(window=30).mean().shift(10)

    # open1分钟移动平均线
    df['open_ma_1m_1'] = df['open'].rolling(window=6).mean()
    df['open_ma_1m_5'] = df['open'].rolling(window=6).mean().shift(5)
    df['open_ma_1m_10'] = df['open'].rolling(window=6).mean().shift(10)
    # open5分钟移动平均线
    df['open_ma_5m_1'] = df['open'].rolling(window=30).mean()
    df['open_ma_5m_5'] = df['open'].rolling(window=30).mean().shift(5)
    df['open_ma_5m_10'] = df['open'].rolling(window=30).mean().shift(10)

    # min1分钟移动平均线
    df['min_ma_1m_1'] = df['min'].rolling(window=6).mean()
    df['min_ma_1m_5'] = df['min'].rolling(window=6).mean().shift(5)
    df['min_ma_1m_10'] = df['min'].rolling(window=6).mean().shift(10)
    # min5分钟移动平均线
    df['min_ma_5m_1'] = df['min'].rolling(window=30).mean()
    df['min_ma_5m_5'] = df['min'].rolling(window=30).mean().shift(5)
    df['min_ma_5m_10'] = df['min'].rolling(window=30).mean().shift(10)

    # max1分钟移动平均线
    df['max_ma_1m_1'] = df['max'].rolling(window=6).mean()
    df['max_ma_1m_5'] = df['max'].rolling(window=6).mean().shift(5)
    df['max_ma_1m_10'] = df['max'].rolling(window=6).mean().shift(10)
    # max5分钟移动平均线
    df['max_ma_5m_1'] = df['max'].rolling(window=30).mean()
    df['max_ma_5m_5'] = df['max'].rolling(window=30).mean().shift(5)
    df['max_ma_5m_10'] = df['max'].rolling(window=30).mean().shift(10)

    # 前值
    df['open_pre_10s'] = df['open'].shift(1)
    df['open_pre_50s'] = df['open'].shift(5)
    df['open_pre_300s'] = df['open'].shift(30)

    df['close_pre_10s'] = df['close'].shift(1)
    df['close_pre_50s'] = df['close'].shift(5)
    df['close_pre_300s'] = df['close'].shift(30)

    df['min_pre_10s'] = df['min'].shift(1)
    df['min_pre_50s'] = df['min'].shift(5)
    df['min_pre_300s'] = df['min'].shift(30)

    df['max_pre_10s'] = df['max'].shift(1)
    df['max_pre_50s'] = df['max'].shift(5)
    df['max_pre_300s'] = df['max'].shift(30)

    df['buy_pre_10s'] = df['buy'].shift(1)
    df['buy_pre_50s'] = df['buy'].shift(5)
    df['buy_pre_300s'] = df['buy'].shift(30)

    df['sell_pre_10s'] = df['sell'].shift(1)
    df['sell_pre_50s'] = df['sell'].shift(5)
    df['sell_pre_300s'] = df['sell'].shift(30)

    df.replace(-np.inf, np.nan)
    df = df[~df.isin([np.nan, np.inf, -np.inf])].astype(np.float64)
    # print(np.isinf(df).any())
    df.dropna(inplace=True)
    return df

if __name__ == '__main__':
    begin = datetime.date(2024,1,8)
    end = datetime.date(2024,1,19)
    data_10S_all = get_data(begin,end)

    df_10S_pre = pd.read_csv("/data/market_maker/origin_data/S10_data_20240107.csv",index_col = 0)
    df = pd.concat([df_10S_pre,data_10S_all])
    df.to_csv("/data/market_maker/origin_data/S10_data.csv")
    df.index = pd.to_datetime(df.index)

    df['volumn'] = df['sell'] + df['buy']
    df['volumn_r'] = df['sell_r'] + df['buy_r']
    df['volumn_count'] = df['sell_count'] + df['buy_count']

    top_1_pct = int(len(df) * 0.01)
    threshold_volumn = df['volumn'].nlargest(top_1_pct).min()
    threshold_sell = df['sell'].nlargest(top_1_pct).min()
    threshold_buy = df['buy'].nlargest(top_1_pct).min()
    threshold_volumn_r = df['volumn_r'].nlargest(top_1_pct).min()
    threshold_buy_r = df['buy_r'].nlargest(top_1_pct).min()
    threshold_sell_r = df['sell_r'].nlargest(top_1_pct).min()
    threshold_volumn_count = df['volumn_count'].nlargest(top_1_pct).min()
    threshold_buy_count = df['buy_count'].nlargest(top_1_pct).min()
    threshold_sell_count = df['sell_count'].nlargest(top_1_pct).min()

    df_result = get_indicators(df)
    # 打印新的DataFrame
    df_result.to_csv("/data/market_maker/label_data/label_indicator_data_new.csv")

import gzip
import json
import pandas as pd
import datetime
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import numpy as np

# Modelling
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint

# Tree Visualisation
from sklearn.tree import export_graphviz
from IPython.display import Image
import graphviz
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from collections import Counter
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


df = pd.read_csv("/data/market_maker/S_data.csv",index_col = 0)
big_break = pd.read_csv('/data/market_maker/big_break1227.csv',index_col = False,encoding='unicode_escape')
# 标记砸盘
df.index = pd.to_datetime(df.index)
big_break['Start Time'] = pd.to_datetime(big_break['Start Time'])
big_break['End Time'] = pd.to_datetime(big_break['End Time'])

# 添加标记列并初始化为 0
df['break'] = 0

# 在时间区间内的时间标记为 1
for index, row in big_break.iterrows():
    start_time = row['Start Time']
    end_time = row['End Time']
    df.loc[(df.index >= start_time) & (df.index < end_time), 'break'] = 1

df.dropna(inplace=True)

# 计算前5分钟'sell'值的和
rolling_sell_sum = df['sell'].rolling('5T').sum()
# sell5min变动----shift 1
df['sell_5m_5m_1'] = (rolling_sell_sum/rolling_sell_sum.shift(300)-1).shift(1)
df['sell_5m_1m_1'] = (rolling_sell_sum/rolling_sell_sum.shift(60)-1).shift(1)
df['sell_5m_10s_1'] = (rolling_sell_sum/rolling_sell_sum.shift(10)-1).shift(1)

# sell5min变动----shift 10
df['sell_5m_5m_10'] = (rolling_sell_sum/rolling_sell_sum.shift(300)-1).shift(10)
df['sell_5m_1m_10'] = (rolling_sell_sum/rolling_sell_sum.shift(60)-1).shift(10)
df['sell_5m_10s_10'] = (rolling_sell_sum/rolling_sell_sum.shift(10)-1).shift(10)

# 计算前1分钟'sell'值的和
rolling_sell_sum_1 = df['sell'].rolling('1T').sum()
# sell1min变动----shift 1
df['sell_1m_5m_1'] = (rolling_sell_sum_1/rolling_sell_sum_1.shift(300)-1).shift(1)
df['sell_1m_1m_1'] = (rolling_sell_sum_1/rolling_sell_sum_1.shift(60)-1).shift(1)
df['sell_1m_10s_1'] = (rolling_sell_sum_1/rolling_sell_sum_1.shift(10)-1).shift(1)

# sell1min变动----shift 10
df['sell_1m_5m_10'] = (rolling_sell_sum_1/rolling_sell_sum_1.shift(300)-1).shift(10)
df['sell_1m_1m_10'] = (rolling_sell_sum_1/rolling_sell_sum_1.shift(60)-1).shift(10)
df['sell_1m_10s_10'] = (rolling_sell_sum_1/rolling_sell_sum_1.shift(10)-1).shift(10)

# 计算前10秒'sell'值的和
rolling_sell_sum_10s = df['sell'].rolling('10S').sum()
# sell10s变动----shift 1
df['sell_10s_5m_1'] = (rolling_sell_sum_10s/rolling_sell_sum_10s.shift(300)-1).shift(1)
df['sell_10s_1m_1'] = (rolling_sell_sum_10s/rolling_sell_sum_10s.shift(60)-1).shift(1)
df['sell_10s_10s_1'] = (rolling_sell_sum_10s/rolling_sell_sum_10s.shift(10)-1).shift(1)

# sell10s变动----shift 10
df['sell_10s_5m_10'] = (rolling_sell_sum_10s/rolling_sell_sum_10s.shift(300)-1).shift(10)
df['sell_10s_1m_10'] = (rolling_sell_sum_10s/rolling_sell_sum_10s.shift(60)-1).shift(10)
df['sell_10s_10s_10'] = (rolling_sell_sum_10s/rolling_sell_sum_10s.shift(10)-1).shift(10)


# 计算sell变动----shift 1
df['sell_change_1s_1'] = df['sell'].diff().shift(1)
df['sell_change_10s_1'] = (df['sell']-df['sell'].shift(10)).shift(1)
df['sell_change_60s_1'] = (df['sell']-df['sell'].shift(60)).shift(1)
df['sell_change_300s_1'] = (df['sell']-df['sell'].shift(300)).shift(1)

# 计算sell变动----shift 10
df['sell_change_1s_10'] = df['sell'].diff().shift(10)
df['sell_change_10s_10'] = (df['sell']-df['sell'].shift(10)).shift(10)
df['sell_change_60s_10'] = (df['sell']-df['sell'].shift(60)).shift(10)
df['sell_change_300s_10'] = (df['sell']-df['sell'].shift(300)).shift(10)


# 计算sell变动率----shift 1
df['sell_change_re_1s_1'] = (df['sell'].diff()/df['sell'].shift(1)).shift(1)
df['sell_change_re_10s_1'] = ((df['sell']-df['sell'].shift(10))/df['sell'].shift(10)).shift(1)
df['sell_change_re_60s_1'] = ((df['sell']-df['sell'].shift(60))/df['sell'].shift(60)).shift(1)
df['sell_change_re_300s_1'] = ((df['sell']-df['sell'].shift(300))/df['sell'].shift(300)).shift(1)

# 计算sell变动率----shift 10
df['sell_change_re_1s_10'] = (df['sell'].diff()/df['sell'].shift(1)).shift(10)
df['sell_change_re_10s_10'] = ((df['sell']-df['sell'].shift(10))/df['sell'].shift(10)).shift(10)
df['sell_change_re_60s_10'] = ((df['sell']-df['sell'].shift(60))/df['sell'].shift(60)).shift(10)
df['sell_change_re_300s_10'] = ((df['sell']-df['sell'].shift(300))/df['sell'].shift(300)).shift(10)

# 计算sell波动率----shift 1
df['sell_volatility_5s_1'] = df['sell'].diff().rolling(window=5).std().shift(1)
df['sell_volatility_20s_1'] = df['sell'].diff().rolling(window=20).std().shift(1)
df['sell_volatility_100s_1'] = df['sell'].diff().rolling(window=100).std().shift(1)
# 计算sell波动率----shift 10
df['sell_volatility_5s_10'] = df['sell'].diff().rolling(window=5).std().shift(10)
df['sell_volatility_20s_10'] = df['sell'].diff().rolling(window=20).std().shift(10)
df['sell_volatility_100s_10'] = df['sell'].diff().rolling(window=100).std().shift(10)


# 计算前5分钟'buy'值的和
rolling_buy_sum = df['buy'].rolling('5T').sum()
# buy5min变动----shift 1
df['buy_5m_5m_1'] = (rolling_buy_sum/rolling_buy_sum.shift(300)-1).shift(1)
df['buy_5m_1m_1'] = (rolling_buy_sum/rolling_buy_sum.shift(60)-1).shift(1)
df['buy_5m_10s_1'] = (rolling_buy_sum/rolling_buy_sum.shift(10)-1).shift(1)

# buy5min变动----shift 10
df['buy_5m_5m_10'] = (rolling_buy_sum/rolling_buy_sum.shift(300)-1).shift(10)
df['buy_5m_1m_10'] = (rolling_buy_sum/rolling_buy_sum.shift(60)-1).shift(10)
df['buy_5m_10s_10'] = (rolling_buy_sum/rolling_buy_sum.shift(10)-1).shift(10)


# 计算前1分钟'sell'值的和
rolling_buy_sum_1 = df['buy'].rolling('1T').sum()
# buy1min变动----shift 1
df['buy_1m_5m_1'] = (rolling_buy_sum_1/rolling_buy_sum_1.shift(300)-1).shift(1)
df['buy_1m_1m_1'] = (rolling_buy_sum_1/rolling_buy_sum_1.shift(60)-1).shift(1)
df['buy_1m_10s_1'] = (rolling_buy_sum_1/rolling_buy_sum_1.shift(10)-1).shift(1)

# buy1min变动----shift 10
df['buy_1m_5m_10'] = (rolling_buy_sum_1/rolling_buy_sum_1.shift(300)-1).shift(10)
df['buy_1m_1m_10'] = (rolling_buy_sum_1/rolling_buy_sum_1.shift(60)-1).shift(10)
df['buy_1m_10s_10'] = (rolling_buy_sum_1/rolling_buy_sum_1.shift(10)-1).shift(10)


# 计算前10秒'buy'值的和
rolling_buy_sum_10s = df['buy'].rolling('10S').sum()
# buy10s变动----shift 1
df['buy_10s_5m_1'] = (rolling_buy_sum_10s/rolling_buy_sum_10s.shift(300)-1).shift(1)
df['buy_10s_1m_1'] = (rolling_buy_sum_10s/rolling_buy_sum_10s.shift(60)-1).shift(1)
df['buy_10s_10s_1'] = (rolling_buy_sum_10s/rolling_buy_sum_10s.shift(10)-1).shift(1)

# buy10s变动----shift 10
df['buy_10s_5m_10'] = (rolling_buy_sum_10s/rolling_buy_sum_10s.shift(300)-1).shift(10)
df['buy_10s_1m_10'] = (rolling_buy_sum_10s/rolling_buy_sum_10s.shift(60)-1).shift(10)
df['buy_10s_10s_10'] = (rolling_buy_sum_10s/rolling_buy_sum_10s.shift(10)-1).shift(10)



# 计算buy变动----shift 1
df['buy_change_1s_1'] = df['buy'].diff().shift(1)
df['buy_change_10s_1'] = (df['buy']-df['buy'].shift(10)).shift(1)
df['buy_change_60s_1'] = (df['buy']-df['buy'].shift(60)).shift(1)
df['buy_change_300s_1'] = (df['buy']-df['buy'].shift(300)).shift(1)

# 计算buy变动----shift 10
df['buy_change_1s_10'] = df['buy'].diff().shift(10)
df['buy_change_10s_10'] = (df['buy']-df['buy'].shift(10)).shift(10)
df['buy_change_60s_10'] = (df['buy']-df['buy'].shift(60)).shift(10)
df['buy_change_300s_10'] = (df['buy']-df['buy'].shift(300)).shift(10)


# 计算buy变动率----shift 1
df['buy_change_re_1s_1'] = (df['buy'].diff()/df['buy'].shift(1)).shift(1)
df['buy_change_re_10s_1'] = ((df['buy']-df['buy'].shift(10))/df['buy'].shift(10)).shift(1)
df['buy_change_re_60s_1'] = ((df['buy']-df['buy'].shift(60))/df['buy'].shift(60)).shift(1)
df['buy_change_re_300s_1'] = ((df['buy']-df['buy'].shift(300))/df['buy'].shift(300)).shift(1)

# 计算buy变动率----shift 10
df['buy_change_re_1s_10'] = (df['buy'].diff()/df['buy'].shift(1)).shift(10)
df['buy_change_re_10s_10'] = ((df['buy']-df['buy'].shift(10))/df['buy'].shift(10)).shift(10)
df['buy_change_re_60s_10'] = ((df['buy']-df['buy'].shift(60))/df['buy'].shift(60)).shift(10)
df['buy_change_re_300s_10'] = ((df['buy']-df['buy'].shift(300))/df['buy'].shift(300)).shift(10)


# 计算buy波动率----shift 1
df['buy_volatility_5s_1'] = df['buy'].diff().rolling(window=5).std().shift(1)
df['buy_volatility_20s_1'] = df['buy'].diff().rolling(window=20).std().shift(1)
df['buy_volatility_100s_1'] = df['buy'].diff().rolling(window=100).std().shift(1)
# 计算buy波动率----shift 10
df['buy_volatility_5s_10'] = df['buy'].diff().rolling(window=5).std().shift(10)
df['buy_volatility_20s_10'] = df['buy'].diff().rolling(window=20).std().shift(10)
df['buy_volatility_100s_10'] = df['buy'].diff().rolling(window=100).std().shift(10)

# 计算10秒buy和的波动率----shift 1
df['buy_volatility_10s_sum_10s_1'] = rolling_buy_sum_10s.diff().rolling(window=10).std().shift(1)
df['buy_volatility_10s_sum_30s_1'] = rolling_buy_sum_10s.diff().rolling(window=30).std().shift(1)
df['buy_volatility_10s_sum_100s_1'] = rolling_buy_sum_10s.diff().rolling(window=100).std().shift(1)
# 计算10秒buy和的波动率----shift 10
df['buy_volatility_10s_sum_10s_10'] = rolling_buy_sum_10s.diff().rolling(window=10).std().shift(10)
df['buy_volatility_10s_sum_30s_10'] = rolling_buy_sum_10s.diff().rolling(window=30).std().shift(10)
df['buy_volatility_10s_sum_100s_10'] = rolling_buy_sum_10s.diff().rolling(window=100).std().shift(10)

# 计算1分钟buy和的波动率----shift 1
df['buy_volatility_1m_sum_10s_1'] = rolling_buy_sum_1.diff().rolling(window=10).std().shift(1)
df['buy_volatility_1m_sum_30s_1'] = rolling_buy_sum_1.diff().rolling(window=30).std().shift(1)
df['buy_volatility_1m_sum_100s_1'] = rolling_buy_sum_1.diff().rolling(window=100).std().shift(1)
# 计算1分钟buy和的波动率----shift 10
df['buy_volatility_1m_sum_10s_10'] = rolling_buy_sum_1.diff().rolling(window=10).std().shift(10)
df['buy_volatility_1m_sum_30s_10'] = rolling_buy_sum_1.diff().rolling(window=30).std().shift(10)
df['buy_volatility_1m_sum_100s_10'] = rolling_buy_sum_1.diff().rolling(window=100).std().shift(10)

# 计算5分钟buy和的波动率----shift 1
df['buy_volatility_5m_sum_10s_1'] = rolling_buy_sum.diff().rolling(window=10).std().shift(1)
df['buy_volatility_5m_sum_30s_1'] = rolling_buy_sum.diff().rolling(window=30).std().shift(1)
df['buy_volatility_5m_sum_100s_1'] = rolling_buy_sum.diff().rolling(window=100).std().shift(1)
# 计算5分钟buy和的波动率----shift 10
df['buy_volatility_5m_sum_10s_10'] = rolling_buy_sum.diff().rolling(window=10).std().shift(10)
df['buy_volatility_5m_sum_30s_10'] = rolling_buy_sum.diff().rolling(window=30).std().shift(10)
df['buy_volatility_5m_sum_100s_10'] = rolling_buy_sum.diff().rolling(window=100).std().shift(10)



# 计算买卖之差占比（市场情绪）----shift 1
df['market_sentiment_1'] = ((df['buy'] - df['sell']) / (df['buy'] + df['sell'])).shift(1)
# 计算买卖之差占比（市场情绪）----shift 10
df['market_sentiment_10'] = ((df['buy'] - df['sell']) / (df['buy'] + df['sell'])).shift(10)
# 计算买卖之差占比（市场情绪）----shift 60
df['market_sentimen0t_60'] = ((df['buy'] - df['sell']) / (df['buy'] + df['sell'])).shift(60)
# 计算买卖之差占比（市场情绪）----shift 120
df['market_sentiment_120'] = ((df['buy'] - df['sell']) / (df['buy'] + df['sell'])).shift(120)

# volumn 
volumn = df['sell'] + df['buy']
volumn_count = df['sell_count'] + df['buy_count']

# 计算前5分钟'volumn'值的和
rolling_volumn_sum = volumn.rolling('5T').sum()
# volumn5min变动----shift 1
df['volumn_5m_5m_1'] = (rolling_volumn_sum/rolling_volumn_sum.shift(300)-1).shift(1)
df['volumn_5m_1m_1'] = (rolling_volumn_sum/rolling_volumn_sum.shift(60)-1).shift(1)
df['volumn_5m_10s_1'] = (rolling_volumn_sum/rolling_volumn_sum.shift(10)-1).shift(1)

# volumn5min变动----shift 10
df['volumn_5m_5m_10'] = (rolling_volumn_sum/rolling_volumn_sum.shift(300)-1).shift(10)
df['volumn_5m_1m_10'] = (rolling_volumn_sum/rolling_volumn_sum.shift(60)-1).shift(10)
df['volumn_5m_10s_10'] = (rolling_volumn_sum/rolling_volumn_sum.shift(10)-1).shift(10)


# 计算前1分钟'sell'值的和
rolling_volumn_sum_1 = volumn.rolling('1T').sum()
# volumn1min变动----shift 1
df['volumn_1m_5m_1'] = (rolling_volumn_sum_1/rolling_volumn_sum_1.shift(300)-1).shift(1)
df['volumn_1m_1m_1'] = (rolling_volumn_sum_1/rolling_volumn_sum_1.shift(60)-1).shift(1)
df['volumn_1m_10s_1'] = (rolling_volumn_sum_1/rolling_volumn_sum_1.shift(10)-1).shift(1)

# volumn1min变动----shift 10
df['volumn_1m_5m_10'] = (rolling_volumn_sum_1/rolling_volumn_sum_1.shift(300)-1).shift(10)
df['volumn_1m_1m_10'] = (rolling_volumn_sum_1/rolling_volumn_sum_1.shift(60)-1).shift(10)
df['volumn_1m_10s_10'] = (rolling_volumn_sum_1/rolling_volumn_sum_1.shift(10)-1).shift(10)


# 计算前10秒'volumn'值的和
rolling_volumn_sum_10s = volumn.rolling('10S').sum()
# volumn10s变动----shift 1
df['volumn_10s_5m_1'] = (rolling_volumn_sum_10s/rolling_volumn_sum_10s.shift(300)-1).shift(1)
df['volumn_10s_1m_1'] = (rolling_volumn_sum_10s/rolling_volumn_sum_10s.shift(60)-1).shift(1)
df['volumn_10s_10s_1'] = (rolling_volumn_sum_10s/rolling_volumn_sum_10s.shift(10)-1).shift(1)

# volumn10s变动----shift 10
df['volumn_10s_5m_10'] = (rolling_volumn_sum_10s/rolling_volumn_sum_10s.shift(300)-1).shift(10)
df['volumn_10s_1m_10'] = (rolling_volumn_sum_10s/rolling_volumn_sum_10s.shift(60)-1).shift(10)
df['volumn_10s_10s_10'] = (rolling_volumn_sum_10s/rolling_volumn_sum_10s.shift(10)-1).shift(10)



# 计算volumn变动----shift 1
df['volumn_change_1s_1'] = volumn.diff().shift(1)
df['volumn_change_10s_1'] = (volumn-volumn.shift(10)).shift(1)
df['volumn_change_60s_1'] = (volumn-volumn.shift(60)).shift(1)
df['volumn_change_300s_1'] = (volumn-volumn.shift(300)).shift(1)

# 计算volumn变动----shift 10
df['volumn_change_1s_10'] = volumn.diff().shift(10)
df['volumn_change_10s_10'] = (volumn-volumn.shift(10)).shift(10)
df['volumn_change_60s_10'] = (volumn-volumn.shift(60)).shift(10)
df['volumn_change_300s_10'] = (volumn-volumn.shift(300)).shift(10)


# 计算volumn变动率----shift 1
df['volumn_change_re_1s_1'] = (volumn.diff()/volumn.shift(1)).shift(1)
df['volumn_change_re_10s_1'] = ((volumn-volumn.shift(10))/volumn.shift(10)).shift(1)
df['volumn_change_re_60s_1'] = ((volumn-volumn.shift(60))/volumn.shift(60)).shift(1)
df['volumn_change_re_300s_1'] = ((volumn-volumn.shift(300))/volumn.shift(300)).shift(1)

# 计算volumn变动率----shift 10
df['volumn_change_re_1s_10'] = (volumn.diff()/volumn.shift(1)).shift(10)
df['volumn_change_re_10s_10'] = ((volumn-volumn.shift(10))/volumn.shift(10)).shift(10)
df['volumn_change_re_60s_10'] = ((volumn-volumn.shift(60))/volumn.shift(60)).shift(10)
df['volumn_change_re_300s_10'] = ((volumn-volumn.shift(300))/volumn.shift(300)).shift(10)


# 计算volumn波动率----shift 1
df['volumn_volatility_5s_1'] = volumn.diff().rolling(window=5).std().shift(1)
df['volumn_volatility_20s_1'] = volumn.diff().rolling(window=20).std().shift(1)
df['volumn_volatility_100s_1'] = volumn.diff().rolling(window=100).std().shift(1)
# 计算volumn波动率----shift 10
df['volumn_volatility_5s_10'] = volumn.diff().rolling(window=5).std().shift(10)
df['volumn_volatility_20s_10'] = volumn.diff().rolling(window=20).std().shift(10)
df['volumn_volatility_100s_10'] = volumn.diff().rolling(window=100).std().shift(10)

# 计算10秒volumn和的波动率----shift 1
df['volumn_volatility_10s_sum_10s_1'] = rolling_volumn_sum_10s.diff().rolling(window=10).std().shift(1)
df['volumn_volatility_10s_sum_30s_1'] = rolling_volumn_sum_10s.diff().rolling(window=30).std().shift(1)
df['volumn_volatility_10s_sum_100s_1'] = rolling_volumn_sum_10s.diff().rolling(window=100).std().shift(1)
# 计算10秒volumn和的波动率----shift 10
df['volumn_volatility_10s_sum_10s_10'] = rolling_volumn_sum_10s.diff().rolling(window=10).std().shift(10)
df['volumn_volatility_10s_sum_30s_10'] = rolling_volumn_sum_10s.diff().rolling(window=30).std().shift(10)
df['volumn_volatility_10s_sum_100s_10'] = rolling_volumn_sum_10s.diff().rolling(window=100).std().shift(10)

# 计算1分钟volumn和的波动率----shift 1
df['volumn_volatility_1m_sum_10s_1'] = rolling_volumn_sum_1.diff().rolling(window=10).std().shift(1)
df['volumn_volatility_1m_sum_30s_1'] = rolling_volumn_sum_1.diff().rolling(window=30).std().shift(1)
df['volumn_volatility_1m_sum_100s_1'] = rolling_volumn_sum_1.diff().rolling(window=100).std().shift(1)
# 计算1分钟volumn和的波动率----shift 10
df['volumn_volatility_1m_sum_10s_10'] = rolling_volumn_sum_1.diff().rolling(window=10).std().shift(10)
df['volumn_volatility_1m_sum_30s_10'] = rolling_volumn_sum_1.diff().rolling(window=30).std().shift(10)
df['volumn_volatility_1m_sum_100s_10'] = rolling_volumn_sum_1.diff().rolling(window=100).std().shift(10)

# 计算5分钟volumn和的波动率----shift 1
df['volumn_volatility_5m_sum_10s_1'] = rolling_volumn_sum.diff().rolling(window=10).std().shift(1)
df['volumn_volatility_5m_sum_30s_1'] = rolling_volumn_sum.diff().rolling(window=30).std().shift(1)
df['volumn_volatility_5m_sum_100s_1'] = rolling_volumn_sum.diff().rolling(window=100).std().shift(1)
# 计算5分钟volumn和的波动率----shift 10
df['volumn_volatility_5m_sum_10s_10'] = rolling_volumn_sum.diff().rolling(window=10).std().shift(10)
df['volumn_volatility_5m_sum_30s_10'] = rolling_volumn_sum.diff().rolling(window=30).std().shift(10)
df['volumn_volatility_5m_sum_100s_10'] = rolling_volumn_sum.diff().rolling(window=100).std().shift(10)
# 计算5分钟volumn和的波动率----shift 5



# 计算前5分钟'volumn_count'值的和
rolling_volumn_count_sum = volumn_count.rolling('5T').sum()
# volumn_count5min变动----shift 1
df['volumn_count_5m_5m_1'] = (rolling_volumn_count_sum/rolling_volumn_count_sum.shift(300)-1).shift(1)
df['volumn_count_5m_1m_1'] = (rolling_volumn_count_sum/rolling_volumn_count_sum.shift(60)-1).shift(1)
df['volumn_count_5m_10s_1'] = (rolling_volumn_count_sum/rolling_volumn_count_sum.shift(10)-1).shift(1)

# volumn_count5min变动----shift 10
df['volumn_count_5m_5m_10'] = (rolling_volumn_count_sum/rolling_volumn_count_sum.shift(300)-1).shift(10)
df['volumn_count_5m_1m_10'] = (rolling_volumn_count_sum/rolling_volumn_count_sum.shift(60)-1).shift(10)
df['volumn_count_5m_10s_10'] = (rolling_volumn_count_sum/rolling_volumn_count_sum.shift(10)-1).shift(10)


# 计算前1分钟'sell'值的和
rolling_volumn_count_sum_1 = volumn_count.rolling('1T').sum()
# volumn_count1min变动----shift 1
df['volumn_count_1m_5m_1'] = (rolling_volumn_count_sum_1/rolling_volumn_count_sum_1.shift(300)-1).shift(1)
df['volumn_count_1m_1m_1'] = (rolling_volumn_count_sum_1/rolling_volumn_count_sum_1.shift(60)-1).shift(1)
df['volumn_count_1m_10s_1'] = (rolling_volumn_count_sum_1/rolling_volumn_count_sum_1.shift(10)-1).shift(1)

# volumn_count1min变动----shift 10
df['volumn_count_1m_5m_10'] = (rolling_volumn_count_sum_1/rolling_volumn_count_sum_1.shift(300)-1).shift(10)
df['volumn_count_1m_1m_10'] = (rolling_volumn_count_sum_1/rolling_volumn_count_sum_1.shift(60)-1).shift(10)
df['volumn_count_1m_10s_10'] = (rolling_volumn_count_sum_1/rolling_volumn_count_sum_1.shift(10)-1).shift(10)


# 计算前10秒'volumn_count'值的和
rolling_volumn_count_sum_10s = volumn_count.rolling('10S').sum()
# volumn_count10s变动----shift 1
df['volumn_count_10s_5m_1'] = (rolling_volumn_count_sum_10s/rolling_volumn_count_sum_10s.shift(300)-1).shift(1)
df['volumn_count_10s_1m_1'] = (rolling_volumn_count_sum_10s/rolling_volumn_count_sum_10s.shift(60)-1).shift(1)
df['volumn_count_10s_10s_1'] = (rolling_volumn_count_sum_10s/rolling_volumn_count_sum_10s.shift(10)-1).shift(1)

# volumn_count10s变动----shift 10
df['volumn_count_10s_5m_10'] = (rolling_volumn_count_sum_10s/rolling_volumn_count_sum_10s.shift(300)-1).shift(10)
df['volumn_count_10s_1m_10'] = (rolling_volumn_count_sum_10s/rolling_volumn_count_sum_10s.shift(60)-1).shift(10)
df['volumn_count_10s_10s_10'] = (rolling_volumn_count_sum_10s/rolling_volumn_count_sum_10s.shift(10)-1).shift(10)



# 计算volumn_count变动----shift 1
df['volumn_count_change_1s_1'] = volumn_count.diff().shift(1)
df['volumn_count_change_10s_1'] = (volumn_count-volumn_count.shift(10)).shift(1)
df['volumn_count_change_60s_1'] = (volumn_count-volumn_count.shift(60)).shift(1)
df['volumn_count_change_300s_1'] = (volumn_count-volumn_count.shift(300)).shift(1)

# 计算volumn_count变动----shift 10
df['volumn_count_change_1s_10'] = volumn_count.diff().shift(10)
df['volumn_count_change_10s_10'] = (volumn_count-volumn_count.shift(10)).shift(10)
df['volumn_count_change_60s_10'] = (volumn_count-volumn_count.shift(60)).shift(10)
df['volumn_count_change_300s_10'] = (volumn_count-volumn_count.shift(300)).shift(10)


# 计算volumn_count变动率----shift 1
df['volumn_count_change_re_1s_1'] = (volumn_count.diff()/volumn_count.shift(1)).shift(1)
df['volumn_count_change_re_10s_1'] = ((volumn_count-volumn_count.shift(10))/volumn_count.shift(10)).shift(1)
df['volumn_count_change_re_60s_1'] = ((volumn_count-volumn_count.shift(60))/volumn_count.shift(60)).shift(1)
df['volumn_count_change_re_300s_1'] = ((volumn_count-volumn_count.shift(300))/volumn_count.shift(300)).shift(1)

# 计算volumn_count变动率----shift 10
df['volumn_count_change_re_1s_10'] = (volumn_count.diff()/volumn_count.shift(1)).shift(10)
df['volumn_count_change_re_10s_10'] = ((volumn_count-volumn_count.shift(10))/volumn_count.shift(10)).shift(10)
df['volumn_count_change_re_60s_10'] = ((volumn_count-volumn_count.shift(60))/volumn_count.shift(60)).shift(10)
df['volumn_count_change_re_300s_10'] = ((volumn_count-volumn_count.shift(300))/volumn_count.shift(300)).shift(10)


# 计算volumn_count波动率----shift 1
df['volumn_count_volatility_5s_1'] = volumn_count.diff().rolling(window=5).std().shift(1)
df['volumn_count_volatility_20s_1'] = volumn_count.diff().rolling(window=20).std().shift(1)
df['volumn_count_volatility_100s_1'] = volumn_count.diff().rolling(window=100).std().shift(1)
# 计算volumn_count波动率----shift 10
df['volumn_count_volatility_5s_10'] = volumn_count.diff().rolling(window=5).std().shift(10)
df['volumn_count_volatility_20s_10'] = volumn_count.diff().rolling(window=20).std().shift(10)
df['volumn_count_volatility_100s_10'] = volumn_count.diff().rolling(window=100).std().shift(10)

# 计算10秒volumn_count和的波动率----shift 1
df['volumn_count_volatility_10s_sum_10s_1'] = rolling_volumn_count_sum_10s.diff().rolling(window=10).std().shift(1)
df['volumn_count_volatility_10s_sum_30s_1'] = rolling_volumn_count_sum_10s.diff().rolling(window=30).std().shift(1)
df['volumn_count_volatility_10s_sum_100s_1'] = rolling_volumn_count_sum_10s.diff().rolling(window=100).std().shift(1)
# 计算10秒volumn_count和的波动率----shift 10
df['volumn_count_volatility_10s_sum_10s_10'] = rolling_volumn_count_sum_10s.diff().rolling(window=10).std().shift(10)
df['volumn_count_volatility_10s_sum_30s_10'] = rolling_volumn_count_sum_10s.diff().rolling(window=30).std().shift(10)
df['volumn_count_volatility_10s_sum_100s_10'] = rolling_volumn_count_sum_10s.diff().rolling(window=100).std().shift(10)

# 计算1分钟volumn_count和的波动率----shift 1
df['volumn_count_volatility_1m_sum_10s_1'] = rolling_volumn_count_sum_1.diff().rolling(window=10).std().shift(1)
df['volumn_count_volatility_1m_sum_30s_1'] = rolling_volumn_count_sum_1.diff().rolling(window=30).std().shift(1)
df['volumn_count_volatility_1m_sum_100s_1'] = rolling_volumn_count_sum_1.diff().rolling(window=100).std().shift(1)
# 计算1分钟volumn_count和的波动率----shift 10
df['volumn_count_volatility_1m_sum_10s_10'] = rolling_volumn_count_sum_1.diff().rolling(window=10).std().shift(10)
df['volumn_count_volatility_1m_sum_30s_10'] = rolling_volumn_count_sum_1.diff().rolling(window=30).std().shift(10)
df['volumn_count_volatility_1m_sum_100s_10'] = rolling_volumn_count_sum_1.diff().rolling(window=100).std().shift(10)

# 计算5分钟volumn_count和的波动率----shift 1
df['volumn_count_volatility_5m_sum_10s_1'] = rolling_volumn_count_sum.diff().rolling(window=10).std().shift(1)
df['volumn_count_volatility_5m_sum_30s_1'] = rolling_volumn_count_sum.diff().rolling(window=30).std().shift(1)
df['volumn_count_volatility_5m_sum_100s_1'] = rolling_volumn_count_sum.diff().rolling(window=100).std().shift(1)
# 计算5分钟volumn_count和的波动率----shift 10
df['volumn_count_volatility_5m_sum_10s_10'] = rolling_volumn_count_sum.diff().rolling(window=10).std().shift(10)
df['volumn_count_volatility_5m_sum_30s_10'] = rolling_volumn_count_sum.diff().rolling(window=30).std().shift(10)
df['volumn_count_volatility_5m_sum_100s_10'] = rolling_volumn_count_sum.diff().rolling(window=100).std().shift(10)


 
# 计算价格变动
price_change_1s = df['min']-df['max'].shift(1)
price_change_10s = df['min']-df['max'].shift(10)
price_change_60s = df['min']-df['max'].shift(60)
price_change_300s = df['min']-df['max'].shift(300)

# 价格涨幅
df['price_rise'] = (price_change_1s/df['max'].shift(1)-1).shift(1)
# 计算min变动----shift 1
df['price_rise_1s_1'] =  (df['min']-df['max'].shift(1)).shift(1)
df['price_rise_10s_1'] = (df['min']-df['max'].shift(10)).shift(1)
df['price_rise_60s_1'] = (df['min']-df['max'].shift(60)).shift(1)
df['price_rise_300s_1'] = (df['min']-df['max'].shift(300)).shift(1)

# 计算min变动----shift 10
df['price_rise_1s_10'] = (df['min']-df['max'].shift(1)).shift(10)
df['price_rise_10s_10'] = (df['min']-df['max'].shift(10)).shift(10)
df['price_rise_60s_10'] = (df['min']-df['max'].shift(60)).shift(10)
df['price_rise_300s_10'] = (df['min']-df['max'].shift(300)).shift(10)


# 计算min变动率----shift 1
df['price_rise_re_1s_1'] = ((df['min']-df['max'].shift(1))/df['max'].shift(1)).shift(1)
df['price_rise_re_10s_1'] = ((df['min']-df['max'].shift(10))/df['max'].shift(10)).shift(1)
df['price_rise_re_60s_1'] = ((df['min']-df['max'].shift(60))/df['max'].shift(60)).shift(1)
df['price_rise_re_300s_1'] = ((df['min']-df['max'].shift(300))/df['max'].shift(300)).shift(1)

# 计算min变动率----shift 10
df['price_rise_re_1s_10'] = ((df['min']-df['max'].shift(1))/df['max'].shift(1)).shift(10)
df['price_rise_re_10s_10'] = ((df['min']-df['max'].shift(10))/df['max'].shift(10)).shift(10)
df['price_rise_re_60s_10'] = ((df['min']-df['max'].shift(60))/df['max'].shift(60)).shift(10)
df['price_rise_re_300s_10'] = ((df['min']-df['max'].shift(300))/df['max'].shift(300)).shift(10)


# 计算1s高低价格波动率----shift 1
df['price_volatility_1_5s_1'] = price_change_1s.rolling(window=5).std().shift(1)
df['price_volatility_1_10s_1'] = price_change_1s.rolling(window=10).std().shift(1)
df['price_volatility_1_30s_1'] = price_change_1s.rolling(window=30).std().shift(1)
df['price_volatility_1_60s_1'] = price_change_1s.rolling(window=60).std().shift(1)
# 计算1s高低价格波动率----shift 10
df['price_volatility_1_5s_10'] = price_change_1s.rolling(window=5).std().shift(10)
df['price_volatility_1_10s_10'] = price_change_1s.rolling(window=10).std().shift(10)
df['price_volatility_1_30s_10'] = price_change_1s.rolling(window=30).std().shift(10)
df['price_volatility_1_60s_10'] = price_change_1s.rolling(window=60).std().shift(10)

# 计算10s高低价格波动率----shift 1
df['price_volatility_10_5s_1'] = price_change_10s.rolling(window=5).std().shift(1)
df['price_volatility_10_10s_1'] = price_change_10s.rolling(window=10).std().shift(1)
df['price_volatility_10_30s_1'] = price_change_10s.rolling(window=30).std().shift(1)
df['price_volatility_10_60s_1'] = price_change_10s.rolling(window=60).std().shift(1)
# 计算10s高低价格波动率----shift 10
df['price_volatility_10_5s_10'] = price_change_10s.rolling(window=5).std().shift(10)
df['price_volatility_10_10s_10'] = price_change_10s.rolling(window=10).std().shift(10)
df['price_volatility_10_30s_10'] = price_change_10s.rolling(window=30).std().shift(10)
df['price_volatility_10_60s_10'] = price_change_10s.rolling(window=60).std().shift(10)

# 计算60s高低价格波动率----shift 1
df['price_volatility_60_5s_1'] = price_change_60s.rolling(window=5).std().shift(1)
df['price_volatility_60_10s_1'] = price_change_60s.rolling(window=10).std().shift(1)
df['price_volatility_60_30s_1'] = price_change_60s.rolling(window=30).std().shift(1)
df['price_volatility_60_60s_1'] = price_change_60s.rolling(window=60).std().shift(1)
# 计算60s高低价格波动率----shift 10
df['price_volatility_60_5s_10'] = price_change_60s.rolling(window=5).std().shift(10)
df['price_volatility_60_10s_10'] = price_change_60s.rolling(window=10).std().shift(10)
df['price_volatility_60_30s_10'] = price_change_60s.rolling(window=30).std().shift(10)
df['price_volatility_60_60s_10'] = price_change_60s.rolling(window=60).std().shift(10)

# 计算300s高低价格波动率----shift 1
df['price_volatility_300_5s_1'] = price_change_300s.rolling(window=5).std().shift(1)
df['price_volatility_300_10s_1'] = price_change_300s.rolling(window=10).std().shift(1)
df['price_volatility_300_30s_1'] = price_change_300s.rolling(window=30).std().shift(1)
df['price_volatility_300_60s_1'] = price_change_300s.rolling(window=60).std().shift(1)
# 计算60s高低价格波动率----shift 10
df['price_volatility_300_5s_10'] = price_change_300s.rolling(window=5).std().shift(10)
df['price_volatility_300_10s_10'] = price_change_300s.rolling(window=10).std().shift(10)
df['price_volatility_300_30s_10'] = price_change_300s.rolling(window=30).std().shift(10)
df['price_volatility_300_60s_10'] = price_change_300s.rolling(window=60).std().shift(10)

# 计算min变动----shift 1
df['min_change_1s_1'] = df['min'].diff().shift(1)
df['min_change_10s_1'] = (df['min']-df['min'].shift(10)).shift(1)
df['min_change_60s_1'] = (df['min']-df['min'].shift(60)).shift(1)
df['min_change_300s_1'] = (df['min']-df['min'].shift(300)).shift(1)

# 计算min变动----shift 10
df['min_change_1s_10'] = df['min'].diff().shift(10)
df['min_change_10s_10'] = (df['min']-df['min'].shift(10)).shift(10)
df['min_change_60s_10'] = (df['min']-df['min'].shift(60)).shift(10)
df['min_change_300s_10'] = (df['min']-df['min'].shift(300)).shift(10)

# 计算min变动率----shift 1
df['min_change_re_1s_1'] = (df['min'].diff()/df['min'].shift(1)).shift(1)
df['min_change_re_10s_1'] = ((df['min']-df['min'].shift(10))/df['min'].shift(10)).shift(1)
df['min_change_re_60s_1'] = ((df['min']-df['min'].shift(60))/df['min'].shift(60)).shift(1)
df['min_change_re_300s_1'] = ((df['min']-df['min'].shift(300))/df['min'].shift(300)).shift(1)

# 计算min变动率----shift 10
df['min_change_re_1s_10'] = (df['min'].diff()/df['min'].shift(1)).shift(10)
df['min_change_re_10s_10'] = ((df['min']-df['min'].shift(10))/df['min'].shift(10)).shift(10)
df['min_change_re_60s_10'] = ((df['min']-df['min'].shift(60))/df['min'].shift(60)).shift(10)
df['min_change_re_300s_10'] = ((df['min']-df['min'].shift(300))/df['min'].shift(300)).shift(10)


# 计算min波动率----shift 1
df['min_volatility_5s_1'] = df['min'].diff().rolling(window=5).std().shift(1)
df['min_volatility_20s_1'] = df['min'].diff().rolling(window=20).std().shift(1)
df['min_volatility_100s_1'] = df['min'].diff().rolling(window=100).std().shift(1)
# 计算min波动率----shift 10
df['min_volatility_5s_10'] = df['min'].diff().rolling(window=5).std().shift(10)
df['min_volatility_20s_10'] = df['min'].diff().rolling(window=20).std().shift(10)
df['min_volatility_100s_10'] = df['min'].diff().rolling(window=100).std().shift(10)

# 计算max变动----shift 1
df['max_change_1s_1'] = df['max'].diff().shift(1)
df['max_change_10s_1'] = (df['max']-df['max'].shift(10)).shift(1)
df['max_change_60s_1'] = (df['max']-df['max'].shift(60)).shift(1)
df['max_change_300s_1'] = (df['max']-df['max'].shift(300)).shift(1)

# 计算max变动----shift 10
df['max_change_1s_10'] = df['max'].diff().shift(10)
df['max_change_10s_10'] = (df['max']-df['max'].shift(10)).shift(10)
df['max_change_60s_10'] = (df['max']-df['max'].shift(60)).shift(10)
df['max_change_300s_10'] = (df['max']-df['max'].shift(300)).shift(10)


# 计算max变动率----shift 1
df['max_change_re_1s_1'] = (df['max'].diff()/df['max'].shift(1)).shift(1)
df['max_change_re_10s_1'] = ((df['max']-df['max'].shift(10))/df['max'].shift(10)).shift(1)
df['max_change_re_60s_1'] = ((df['max']-df['max'].shift(60))/df['max'].shift(60)).shift(1)
df['max_change_re_300s_1'] = ((df['max']-df['max'].shift(300))/df['max'].shift(300)).shift(1)

# 计算max变动率----shift 10
df['max_change_re_1s_10'] = (df['max'].diff()/df['max'].shift(1)).shift(10)
df['max_change_re_10s_10'] = ((df['max']-df['max'].shift(10))/df['max'].shift(10)).shift(10)
df['max_change_re_60s_10'] = ((df['max']-df['max'].shift(60))/df['max'].shift(60)).shift(10)
df['max_change_re_300s_10'] = ((df['max']-df['max'].shift(300))/df['max'].shift(300)).shift(10)


# 计算max波动率----shift 1
df['max_volatility_5s_1'] = df['max'].diff().rolling(window=5).std().shift(1)
df['max_volatility_20s_1'] = df['max'].diff().rolling(window=20).std().shift(1)
df['max_volatility_100s_1'] = df['max'].diff().rolling(window=100).std().shift(1)
# 计算max波动率----shift 10
df['max_volatility_5s_10'] = df['max'].diff().rolling(window=5).std().shift(10)
df['max_volatility_20s_10'] = df['max'].diff().rolling(window=20).std().shift(10)
df['max_volatility_100s_10'] = df['max'].diff().rolling(window=100).std().shift(10)

# 价格区间
price_range = df['min']-df['max']
df['price_range_1'] = price_range.shift(1)
df['price_range_10'] = price_range.shift(10)
df['price_range_60'] = price_range.shift(60)

# 计算price_range变动----shift 1
df['price_range_change_1s_1'] = price_range.diff().shift(1)
df['price_range_change_10s_1'] = (price_range-price_range.shift(10)).shift(1)
df['price_range_change_60s_1'] = (price_range-price_range.shift(60)).shift(1)
df['price_range_change_300s_1'] = (price_range-price_range.shift(300)).shift(1)

# 计算price_range变动----shift 10
df['price_range_change_1s_10'] = price_range.diff().shift(10)
df['price_range_change_10s_10'] = (price_range-price_range.shift(10)).shift(10)
df['price_range_change_60s_10'] = (price_range-price_range.shift(60)).shift(10)
df['price_range_change_300s_10'] = (price_range-price_range.shift(300)).shift(10)

# 计算price_range变动率----shift 1
df['price_range_change_re_1s_1'] = (price_range.diff()/price_range.shift(1)).shift(1)
df['price_range_change_re_10s_1'] = ((price_range-price_range.shift(10))/price_range.shift(10)).shift(1)
df['price_range_change_re_60s_1'] = ((price_range-price_range.shift(60))/price_range.shift(60)).shift(1)
df['price_range_change_re_300s_1'] = ((price_range-price_range.shift(300))/price_range.shift(300)).shift(1)

# 计算price_range变动率----shift 10
df['price_range_change_re_1s_10'] = (price_range.diff()/price_range.shift(1)).shift(10)
df['price_range_change_re_10s_10'] = ((price_range-price_range.shift(10))/price_range.shift(10)).shift(10)
df['price_range_change_re_60s_10'] = ((price_range-price_range.shift(60))/price_range.shift(60)).shift(10)
df['price_range_change_re_300s_10'] = ((price_range-price_range.shift(300))/price_range.shift(300)).shift(10)


# 计算price_range波动率----shift 1
df['price_range_volatility_5s_1'] = price_range.diff().rolling(window=5).std().shift(1)
df['price_range_volatility_20s_1'] = price_range.diff().rolling(window=20).std().shift(1)
df['price_range_volatility_100s_1'] = price_range.diff().rolling(window=100).std().shift(1)
# 计算price_range波动率----shift 10
df['price_range_volatility_5s_10'] = price_range.diff().rolling(window=5).std().shift(10)
df['price_range_volatility_20s_10'] = price_range.diff().rolling(window=20).std().shift(10)
df['price_range_volatility_100s_10'] = price_range.diff().rolling(window=100).std().shift(10)


# min1分钟移动平均线
df['min_ma_1m_1'] = df['min'].rolling(window=60).mean().shift(1)
df['min_ma_1m_5'] = df['min'].rolling(window=60).mean().shift(5)
df['min_ma_1m_10'] = df['min'].rolling(window=60).mean().shift(10)
df['min_ma_1m_60'] = df['min'].rolling(window=60).mean().shift(60)
# min10s移动平均线
df['min_ma_10s_1'] = df['min'].rolling(window=10).mean().shift(1)
df['min_ma_10s_5'] = df['min'].rolling(window=10).mean().shift(5)
df['min_ma_10s_10'] = df['min'].rolling(window=10).mean().shift(10)
df['min_ma_10s_60'] = df['min'].rolling(window=10).mean().shift(60)
# min5分钟移动平均线
df['min_ma_5m_1'] = df['min'].rolling(window=300).mean().shift(1)
df['min_ma_5m_5'] = df['min'].rolling(window=300).mean().shift(5)
df['min_ma_5m_10'] = df['min'].rolling(window=300).mean().shift(10)
df['min_ma_5m_60'] = df['min'].rolling(window=300).mean().shift(60)


# max1分钟移动平均线
df['max_ma_1m_1'] = df['max'].rolling(window=60).mean().shift(1)
df['max_ma_1m_5'] = df['max'].rolling(window=60).mean().shift(5)
df['max_ma_1m_10'] = df['max'].rolling(window=60).mean().shift(10)
df['max_ma_1m_60'] = df['max'].rolling(window=60).mean().shift(60)
# max10s移动平均线
df['max_ma_10s_1'] = df['max'].rolling(window=10).mean().shift(1)
df['max_ma_10s_5'] = df['max'].rolling(window=10).mean().shift(5)
df['max_ma_10s_10'] = df['max'].rolling(window=10).mean().shift(10)
df['max_ma_10s_60'] = df['max'].rolling(window=10).mean().shift(60)
# max5分钟移动平均线
df['max_ma_5m_1'] = df['max'].rolling(window=300).mean().shift(1)
df['max_ma_5m_5'] = df['max'].rolling(window=300).mean().shift(5)
df['max_ma_5m_10'] = df['max'].rolling(window=300).mean().shift(10)
df['max_ma_5m_60'] = df['max'].rolling(window=300).mean().shift(60)

# 前值
df['open_pre_1s'] = df['open'].shift(1)
df['open_pre_5s'] = df['open'].shift(5)
df['open_pre_10s'] = df['open'].shift(10)
df['open_pre_60s'] = df['open'].shift(60)

df['close_pre_1s'] = df['close'].shift(1)
df['close_pre_5s'] = df['close'].shift(5)
df['close_pre_10s'] = df['close'].shift(10)
df['close_pre_60s'] = df['close'].shift(60)

df['min_pre_1s'] = df['min'].shift(1)
df['min_pre_5s'] = df['min'].shift(5)
df['min_pre_10s'] = df['min'].shift(10)
df['min_pre_60s'] = df['min'].shift(60)

df['max_pre_1s'] = df['max'].shift(1)
df['max_pre_5s'] = df['max'].shift(5)
df['max_pre_10s'] = df['max'].shift(10)
df['max_pre_60s'] = df['max'].shift(60)

df['buy_pre_1s'] = df['buy'].shift(1)
df['buy_pre_5s'] = df['buy'].shift(5)
df['buy_pre_10s'] = df['buy'].shift(10)
df['buy_pre_60s'] = df['buy'].shift(60)

df['sell_pre_1s'] = df['sell'].shift(1)
df['sell_pre_5s'] = df['sell'].shift(5)
df['sell_pre_10s'] = df['sell'].shift(10)
df['sell_pre_60s'] = df['sell'].shift(60)

df.replace(-np.inf, np.nan)
df = df[~df.isin([np.nan, np.inf, -np.inf])].astype(np.float64)
print(np.isinf(df).any())
df.dropna(inplace=True)

# Data Processing


X = df.drop(columns = ['open','close','max','min','sell','sell_r','sell_count','buy','buy_r','buy_count','break'],axis = 1,inplace = False)
y = df['break']
# Split the data into training and test sets
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
rf = RandomForestClassifier()

# 随机欠采样
rus = RandomUnderSampler(random_state=0,sampling_strategy=0.01)

X_resampled_u1, y_resampled_u1 = rus.fit_resample(X, y)
counter_resampled_u1 = Counter(y_resampled_u1)
print("随机欠采样结果:\n", counter_resampled_u1)
X_train, X_test, y_train, y_test = train_test_split(X_resampled_u1, y_resampled_u1, test_size=0.2)

rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
# 超参数调优
param_dist = {'n_estimators': randint(50,500),
              'max_depth': randint(1,20)}

# Create a random forest classifier
rf1 = RandomForestClassifier(warm_start=True)

# Use random search to find the best hyperparameters
rand_search = RandomizedSearchCV(rf1, 
                                 param_distributions = param_dist, 
                                 n_iter=5, 
                                 cv=5)

# Fit the random search object to the data
rand_search.fit(X_train, y_train)
# Create a variable for the best model
best_rf = rand_search.best_estimator_

# Print the best hyperparameters
print('Best hyperparameters:',  rand_search.best_params_)

# Create a series containing feature importances from the model and feature names from the training data
feature_importances = pd.Series(best_rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)

# Plot a simple bar chart
feature_importances.plot.bar();

y_pred_all = best_rf.predict(X)
accuracy = accuracy_score(y, y_pred_all)
precision = precision_score(y, y_pred_all)
recall = recall_score(y, y_pred_all)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)

pred_y = pd.DataFrame({'y':y,'pred':y_pred_all})
pred_y[pred_y['pred'] == 1].to_csv('/data/market_maker/pred_S_model.csv')
import pandas as pd
import datetime
import numpy as np

df = pd.read_csv("/data/market_maker/origin_data/S_data.csv",index_col = 0)
df.index = pd.to_datetime(df.index)
# drop_events_df = pd.DataFrame(columns=["End Time", "End Min Value", "Fall"])
# drop_events_list = []
# 计算十分钟滚动窗口内的最大最小值
rolling_max = df['max'].rolling("5T").max()
rolling_min = df['min'].rolling("5T").min()

# 计算跌幅
df['price_fall_5min'] = (df['min'] - rolling_max) / rolling_max * 100
df['price_rise_5min'] = (df['max'] - rolling_min) / rolling_min * 100
# 找出5分钟内涨跌幅超过3%的时间段
drop_events = df[(df['price_fall_5min'] <= -3) | (df['price_rise_5min'] >= 3)]

# 罗列砸盘开始和结束时间、开始和结束的"min"值，以及跌幅，并存入新的DataFrame
#for idx in drop_events.index:
    #end_time = idx
    #min_value_end = df.loc[end_time, 'min']
    #fall = price_fall.loc[idx]

    # 将数据存入新的DataFrame
    #drop_events_list.append({"End Time": end_time,  "End Min Value": min_value_end, "Fall": fall})

# 打印新的DataFrame
#drop_events_df = pd.DataFrame(drop_events_list)
drop_events.to_csv("/data/market_maker/label_data/break_5_3_fall_rise.csv")
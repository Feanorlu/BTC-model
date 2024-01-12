"""
Filename: data_test.py
Author: Lu Sumeng
"""
import gzip
import json
import pandas as pd
import datetime
# 读取gzip压缩的JSON文件
#input_gz_file = 'C:/Users/13365/Downloads/2021-01-03.json.gz'
#output_excel_file = 'C:/Users/13365/output.xlsx'

def to_df(date):
    data = []
    file = f'/tt_data/lcheng_v2/tardis/binance-futures/BTCUSDT/trade/{date.strftime("%Y-%m-%d")}.json.gz'    
    with gzip.open(file, 'rb') as f:
        for line in f:
            data.append(json.loads(line))
    # 将数据转换为DataFrame
    df = pd.DataFrame(data)
    # 将时间戳转换为日期时间格式
    df['t'] = pd.to_datetime(df['t'])
    # 将时间戳列设置为索引
    df.set_index('t', inplace=True)
    return df
    
def to_minute_data(df):
    df_s = df[df['s']=='s']
    df_b = df[df['s']=='b']
    resampled_data = df.resample('1T').agg({
    'r': ['first', 'last','max','min'],   
    })
    # 重命名列标题
    resampled_data.columns = ['open', 'close', 'max','min']

    resampled_data_s = df_s.resample('1T').agg({
        'a': 'sum',               # 合计数量
        'aq': 'sum',              # 合计金额
        's': 'count',  # 卖出次数
    })
    # 重命名列标题
    resampled_data_s.columns = ['sell', 'sell_r','sell_count']
    
    resampled_data_b = df_b.resample('1T').agg({
        'a': 'sum',               # 合计数量
        'aq': 'sum',              # 合计金额
        's': 'count',  # 卖出次数
    })
    # 重命名列标题
    resampled_data_b.columns = ['buy', 'buy_r','buy_count']
    
    data_min = resampled_data.join(resampled_data_s).join(resampled_data_b)
    return data_min

def to_10S_data(df):
    df_s = df[df['s']=='s']
    df_b = df[df['s']=='b']
    resampled_data = df.resample('10S').agg({
    'r': ['first', 'last','max','min'],   
    })
    # 重命名列标题
    resampled_data.columns = ['open', 'close', 'max','min']

    resampled_data_s = df_s.resample('10S').agg({
        'a': 'sum',               # 合计数量
        'aq': 'sum',              # 合计金额
        's': 'count',  # 卖出次数
    })
    # 重命名列标题
    resampled_data_s.columns = ['sell', 'sell_r','sell_count']
    
    resampled_data_b = df_b.resample('10S').agg({
        'a': 'sum',               # 合计数量
        'aq': 'sum',              # 合计金额
        's': 'count',  # 卖出次数
    })
    # 重命名列标题
    resampled_data_b.columns = ['buy', 'buy_r','buy_count']
    
    data_10S = resampled_data.join(resampled_data_s).join(resampled_data_b)
    return data_10S

def to_5S_data(df):
    df_s = df[df['s']=='s']
    df_b = df[df['s']=='b']
    resampled_data = df.resample('5S').agg({
    'r': ['first', 'last','max','min'],   
    })
    # 重命名列标题
    resampled_data.columns = ['open', 'close', 'max','min']

    resampled_data_s = df_s.resample('5S').agg({
        'a': 'sum',               # 合计数量
        'aq': 'sum',              # 合计金额
        's': 'count',  # 卖出次数
    })
    # 重命名列标题
    resampled_data_s.columns = ['sell', 'sell_r','sell_count']
    
    resampled_data_b = df_b.resample('5S').agg({
        'a': 'sum',               # 合计数量
        'aq': 'sum',              # 合计金额
        's': 'count',  # 卖出次数
    })
    # 重命名列标题
    resampled_data_b.columns = ['buy', 'buy_r','buy_count']
    
    data_5S = resampled_data.join(resampled_data_s).join(resampled_data_b)
    return data_5S

def to_S_data(df):
    df_s = df[df['s']=='s']
    df_b = df[df['s']=='b']
    resampled_data = df.resample('1S').agg({
    'r': ['first', 'last','max','min'],   
    })
    # 重命名列标题
    resampled_data.columns = ['open', 'close', 'max','min']

    resampled_data_s = df_s.resample('1S').agg({
        'a': 'sum',               # 合计数量
        'aq': 'sum',              # 合计金额
        's': 'count',  # 卖出次数
    })
    # 重命名列标题
    resampled_data_s.columns = ['sell', 'sell_r','sell_count']
    
    resampled_data_b = df_b.resample('1S').agg({
        'a': 'sum',               # 合计数量
        'aq': 'sum',              # 合计金额
        's': 'count',  # 卖出次数
    })
    # 重命名列标题
    resampled_data_b.columns = ['buy', 'buy_r','buy_count']
    
    data_S = resampled_data.join(resampled_data_s).join(resampled_data_b)
    return data_S

def to_hour_data(data_min):
    data_hour = data_min.resample('1h').agg({
    'open': 'first',           
    'close': 'last',  
    'max': 'max',
    'min': 'min',
    'sell': 'sum',  
    'sell_r': 'sum',  
    'sell_count': 'sum',  
    'buy': 'sum',  
    'buy_r': 'sum',  
    'buy_count': 'sum',  
    })
    return data_hour

def get_data(begin,end):
    d = begin
    delta = datetime.timedelta(days=1)
    # data_all = pd.DataFrame()
    # data_min_all = pd.DataFrame()
    data_10S_all = pd.DataFrame()
    # data_S_all = pd.DataFrame()
    # data_5S_all = pd.DataFrame()
    while d <= end:
        df_day = to_df(d)
        # data_all = pd.concat([data_all,df_day])
        # S_day = to_S_data(df_day)
        # data_S_all = pd.concat([data_S_all,S_day])
        # min_day = to_minute_data(df_day)
        # data_min_all = pd.concat([data_min_all,min_day])
        S10_day = to_10S_data(df_day)
        data_10S_all = pd.concat([data_10S_all,S10_day])
        # S5_day = to_5S_data(df_day)
        # data_5S_all = pd.concat([data_5S_all,S5_day])
        d += delta
    return data_10S_all

begin = datetime.date(2023,12,13)
end = datetime.date(2024,1,7)
data_10S_all = get_data(begin,end)


# data_min_all,data_10S_all = get_data(begin,end)
# data_S_all.to_csv("/data/market_maker/S_data.csv")
# data_all.to_csv("/data/market_maker/merge_data.csv")
# data_min_all.to_csv("/data/market_maker/minute_data.csv")


df_10S_pre = pd.read_csv("/data/market_maker/origin_data/S10_data.csv",index_col = 0)
data_10S_all = pd.concat([df_10S_pre,data_10S_all])
data_10S_all.to_csv("/data/market_maker/origin_data/S10_data_20240107.csv")
# data_5S_all.to_csv("/data/market_maker/S5_data.csv")

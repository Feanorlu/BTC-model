import joblib
import requests
import time
import pandas as pd
import get_label_indicator


model_all = joblib.load("/data/market_maker/model/model_all_1pct_20240107.pkl")
model_volumn_all = joblib.load("/data/market_maker/model/model_volumn_all_1pct_20240107.pkl")
model_buy_all = joblib.load("/data/market_maker/model/model_buy_all_1pct_20240107.pkl")
model_sell_all = joblib.load("/data/market_maker/model/model_sell_all_1pct_20240107.pkl")

df_8day = pd.read_csv("/data/market_maker/origin_data/S10_data_20240115.csv",index_col = 0)
df_8day.index = pd.to_datetime(df_8day.index)
df_indicator = get_label_indicator.get_indicators(df_8day)

# 进行分类预测
all_pred = model_all.predict(df_indicator.drop(columns = ['all_1pct'],axis = 1,inplace = False))
volumn_all_pred = model_volumn_all.predict(df_indicator.drop(columns = ['volumn_all_1pct'],axis = 1,inplace = False))
buy_all_pred = model_buy_all.predict(df_indicator.drop(columns = ['buy_all_1pct'],axis = 1,inplace = False))
sell_all_pred = model_sell_all.predict(df_indicator.drop(columns = ['sell_all_1pct'],axis = 1,inplace = False))

pred_y = pd.DataFrame({'all_1pct':df_indicator['all_1pct'],'volumn_all_1pct':df_indicator['volumn_all_1pct'],'buy_all_1pct':df_indicator['buy_all_1pct'],
    'sell_all_1pct':df_indicator['sell_all_1pct'],'all_pred':all_pred,'volumn_all_pred':volumn_all_pred,
    'buy_all_pred':buy_all_pred,'sell_all_pred':sell_all_pred})
pred_y.to_csv('/data/market_maker/model_pred_result_data/pred_all_1pct_20240115.csv')
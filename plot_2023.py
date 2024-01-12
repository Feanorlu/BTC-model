import gzip
import json
import pandas as pd
import datetime
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import joblib

df = pd.read_csv("/data/market_maker/label_data/label_indicator_data_20240107.csv",index_col = 0)
df_2023 = df["2023-01-01":]
rf_all = joblib.load("/data/market_maker/model/model_all_1pct.pkl")
x_2023 = df_2023.drop(columns = ['all_1pct'],axis = 1,inplace = False)
y_2023 = pd.DataFrame(rf_all.predict(x_2023)).shift(2)


# 创建一个带有0-1变量的曲线图
plt.figure(figsize=(10, 6))
plt.step(y_2023.index, y_2023, where='mid')
plt.plot(x_2023.index, x_2023['volumn'],linewidth=0.3)
plt.plot(x_2023.index, x_2023['min'],linewidth=0.3)
plt.xlabel('Time')
plt.ylabel('Binary Variable')
plt.title('Time Series of Binary Variable')
plt.show()
plt.savefig('/data/market_maker/plot/pred_2023_all.png')

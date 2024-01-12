import gzip
import json
import joblib
import pandas as pd
import datetime
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import numpy as np
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

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


df = pd.read_csv("/data/market_maker/label_data/label_indicator_data.csv",index_col = 0)
big_break = pd.read_csv('/data/market_maker/label_data/big_break1227.csv',index_col = False,encoding='unicode_escape')


# 标记砸盘
df.index = pd.to_datetime(df.index)
big_break['Start Time'] = pd.to_datetime(big_break['Start Time'])
big_break['End Time'] = pd.to_datetime(big_break['End Time'])
big_break['Start 10s'] = pd.to_datetime(big_break['Start 10s'])

# 添加标记列并初始化为 0
df['break'] = 0

# 在时间区间内的时间标记为 1
for index, row in big_break.iterrows():
    start_time = row['Start 10s']
    end_time = row['End Time']
    df.loc[(df.index >= start_time) & (df.index < end_time), 'break'] = 1

df.dropna(inplace=True)
df1 = df[:'2023-01-01']
df2 = df['2023-01-01':]

# Data Processing


X_10s = df1.drop(columns = ['break'],axis = 1,inplace = False)[:-2]
y_10s = df1['break'].shift(-2)[:-2]
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_10s, y_10s, test_size=0.2)
rf_10s = RandomForestClassifier(max_depth=18, n_estimators=322)

# 随机欠采样
# rus_10s = RandomUnderSampler(random_state=0,sampling_strategy=0.001)

# X_resampled_10s, y_resampled_10s = rus_10s.fit_resample(X_10s, y_10s)
# counter_resampled_10s = Counter(y_resampled_10s)
# print("随机欠采样结果:\n", counter_resampled_10s)
# X_train_10s, X_test_10s, y_train_10s, y_test_10s = train_test_split(X_resampled_10s, y_resampled_10s, test_size=0.4)

rf_10s.fit(X_train, y_train)
y_pred = rf_10s.predict(X_test)
accuracy_10s_1 = accuracy_score(y_test, y_pred)
precision_10s_1 = precision_score(y_test, y_pred)
recall_10s_1 = recall_score(y_test, y_pred)

print("Accuracy:", accuracy_10s_1)
print("Precision:", precision_10s_1)
print("Recall:", recall_10s_1)
# 超参数调优
#param_dist_10s = {'n_estimators': randint(50,500),
              #'max_depth': randint(1,20)}

# Create a random forest classifier
#rf1_10s = RandomForestClassifier(warm_start=True)

# Use random search to find the best hyperparameters
#rand_search_10s = RandomizedSearchCV(rf1_10s, 
                                 #param_distributions = param_dist_10s, 
                                 #n_iter=5, 
                                 #cv=5)

# Fit the random search object to the data
#rand_search_10s.fit(X_train_10s, y_train_10s)
# Create a variable for the best model
#best_rf_10s = rand_search_10s.best_estimator_

# Print the best hyperparameters
#print('Best hyperparameters:',  rand_search_10s.best_params_)

# Create a series containing feature importances from the model and feature names from the training data
feature_importances_10s_all = pd.Series(rf_10s.feature_importances_, index=X_train.columns).sort_values(ascending=False)
feature_importances_10s_all.to_csv('/data/market_maker/model/indicators_importance_break_all.csv')
# Plot a simple bar chart
#feature_importances_10s.plot.bar();


# Save the model
joblib.dump(rf_10s, "/data/market_maker/model/model_origin_all.pkl")

y_pred_all_10s = rf_10s.predict(X_10s)
accuracy_10s_2 = accuracy_score(y_10s, y_pred_all_10s)
precision_10s_2 = precision_score(y_10s, y_pred_all_10s)
recall_10s_2 = recall_score(y_10s, y_pred_all_10s)
print("2021-2022预测:")
print("Accuracy:", accuracy_10s_2)
print("Precision:", precision_10s_2)
print("Recall:", recall_10s_2)


X_10s_2023 = df2.drop(columns = ['break'],axis = 1,inplace = False)[:-2]
y_10s_2023 = df2['break'].shift(-2)[:-2]
y_pred_2023 = rf_10s.predict(X_10s_2023)
accuracy_10s_3 = accuracy_score(y_10s_2023, y_pred_2023)
precision_10s_3 = precision_score(y_10s_2023, y_pred_2023)
recall_10s_3 = recall_score(y_10s_2023, y_pred_2023)
print("2023预测:")
print("Accuracy:", accuracy_10s_3)
print("Precision:", precision_10s_3)
print("Recall:", recall_10s_3)


pred_y_10s_2023 = pd.DataFrame({'y':y_10s_2023,'pred':y_pred_2023})
pred_y_10s_2023[(pred_y_10s_2023['pred'] == 1)|(pred_y_10s_2023['y'] == 1)].to_csv('/data/market_maker/model_pred_result_data/pred_10S_pred_or_real_2023_all.csv')


X_10s_all = df.drop(columns = ['break'],axis = 1,inplace = False)[:-2]
y_10s_all = df['break'].shift(-2)[:-2]
y_pred_all = rf_10s.predict(X_10s_all)
accuracy_10s_4 = accuracy_score(y_10s_all, y_pred_all)
precision_10s_4 = precision_score(y_10s_all, y_pred_all)
recall_10s_4 = recall_score(y_10s_all, y_pred_all)
print("2021-2023预测:")
print("Accuracy:", accuracy_10s_4)
print("Precision:", precision_10s_4)
print("Recall:", recall_10s_4)


pred_y_10s_all = pd.DataFrame({'y':y_10s_all,'pred':y_pred_all})
pred_y_10s_all[(pred_y_10s_all['pred'] == 1)|(pred_y_10s_all['y'] == 1)].to_csv('/data/market_maker/model_pred_result_data/pred_10S_pred_or_real_all.csv')

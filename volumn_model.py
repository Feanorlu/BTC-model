
import pandas as pd
import datetime
import joblib

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


df = pd.read_csv("/data/market_maker/label_data/label_indicator_data_20240107.csv",index_col = 0)
#df1 = df[:'2023-01-01']
#df2 = df['2023-01-01':]
# Data Processing


X1 = df.drop(columns = ['all_1pct'],axis = 1,inplace = False)[:-2]
y1 = df['all_1pct'].shift(-2)[:-2]
# Split the data into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.2)
rf = RandomForestClassifier(max_depth=18, n_estimators=322)

# 随机欠采样
# rus = RandomUnderSampler(random_state=0,sampling_strategy=0.001)

# X_resampled, y_resampled = rus.fit_resample(X, y)
# counter_resampled = Counter(y_resampled)
# print("随机欠采样结果:\n", counter_resampled)
# X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.4)

rf.fit(X1, y1)
#y_pred1 = rf.predict(X_test)
#accuracy_1 = accuracy_score(y_test, y_pred1)
#precision_1 = precision_score(y_test, y_pred1)
#recall_1 = recall_score(y_test, y_pred1)

#print("Accuracy1:", accuracy_1)
#print("Precision1:", precision_1)
#print("Recall1:", recall_1)
# 超参数调优
#param_dist = {'n_estimators': randint(50,500),
              #'max_depth': randint(1,20)}

# Create a random forest classifier
#rf1 = RandomForestClassifier(warm_start=True)

# Use random search to find the best hyperparameters
#rand_search = RandomizedSearchCV(rf1, 
                                 #param_distributions = param_dist, 
                                 #n_iter=5, 
                                 #cv=5)

# Fit the random search object to the data
#rand_search.fit(X_train, y_train)
# Create a variable for the best model
#best_rf = rand_search.best_estimator_

# Print the best hyperparameters
#print('Best hyperparameters:',  rand_search.best_params_)

# Create a series containing feature importances from the model and feature names from the training data
feature_importances = pd.Series(rf.feature_importances_, index=X1.columns).sort_values(ascending=False)
feature_importances.to_csv('/data/market_maker/model/indicators_importance_all_20240107.csv')

# Plot a simple bar chart
#feature_importances_10s.plot.bar();


# 将模型保存为文件
joblib.dump(rf, "/data/market_maker/model/model_all_1pct_20240107.pkl")

# 预测2023
#X2 = df2.drop(columns = ['all_1pct'],axis = 1,inplace = False)[:-2]
#y2 = df2['all_1pct'].shift(-2)[:-2]

#y_pred2 = rf.predict(X2)
#accuracy_2 = accuracy_score(y2, y_pred2)
#precision_2 = precision_score(y2, y_pred2)
#recall_2 = recall_score(y2, y_pred2)

#print("Accuracy2:", accuracy_2)
#print("Precision2:", precision_2)
#print("Recall2:", recall_2)

#pred_y2 = pd.DataFrame({'y':y2,'pred':y_pred2})
#pred_y2[(pred_y2['pred'] == 1)|(pred_y2['y'] == 1)].to_csv('/data/market_maker/model_pred_result_data/pred_all_1pct_2023.csv')

# 全数据集预测
#X = df.drop(columns = ['all_1pct'],axis = 1,inplace = False)[:-2]
#y = df['all_1pct'].shift(-2)[:-2]

#y_pred = rf.predict(X)
#accuracy = accuracy_score(y, y_pred)
#precision = precision_score(y, y_pred)
#recall = recall_score(y, y_pred)

#print("Accuracy:", accuracy)
#print("Precision:", precision)
#print("Recall:", recall)

#pred_y = pd.DataFrame({'y':y,'pred':y_pred})
#pred_y[(pred_y['pred'] == 1)|(pred_y['y'] == 1)].to_csv('/data/market_maker/model_pred_result_data/pred_all_1pct.csv')
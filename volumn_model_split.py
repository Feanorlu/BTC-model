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


X1 = df.drop(columns = ['volumn_all_1pct'],axis = 1,inplace = False)[:-2]
y1 = df['volumn_all_1pct'].shift(-2)[:-2]
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
feature_importances.to_csv('/data/market_maker/model/indicators_importance_volumn_all_20240107.csv')

# Plot a simple bar chart
#feature_importances_10s.plot.bar();


# 将模型保存为文件
joblib.dump(rf, "/data/market_maker/model/model_volumn_all_1pct_20240107.pkl")



X2 = df.drop(columns = ['sell_all_1pct'],axis = 1,inplace = False)[:-2]
y2 = df['sell_all_1pct'].shift(-2)[:-2]
rf2 = RandomForestClassifier(max_depth=18, n_estimators=322)
rf2.fit(X2, y2)

feature_importances2 = pd.Series(rf2.feature_importances_, index=X2.columns).sort_values(ascending=False)
feature_importances2.to_csv('/data/market_maker/model/indicators_importance_sell_all_20240107.csv')
joblib.dump(rf2, "/data/market_maker/model/model_sell_all_1pct_20240107.pkl")


X3 = df.drop(columns = ['buy_all_1pct'],axis = 1,inplace = False)[:-2]
y3 = df['buy_all_1pct'].shift(-2)[:-2]
rf3 = RandomForestClassifier(max_depth=18, n_estimators=322)
rf3.fit(X3, y3)

feature_importances3 = pd.Series(rf3.feature_importances_, index=X3.columns).sort_values(ascending=False)
feature_importances3.to_csv('/data/market_maker/model/indicators_importance_buy_all_20240107.csv')
joblib.dump(rf3, "/data/market_maker/model/model_buy_all_1pct_20240107.pkl")
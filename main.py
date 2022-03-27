#Importing libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#Importing dataset
df_train = pd.read_csv('train.csv')
df_center_info = pd.read_csv('fulfilment_center.csv')
df_meal_info = pd.read_csv('meal_info.csv')
df_test = pd.read_csv('test.csv')

#Merging dataset
# Merge the training data with the branch and meal information.
df_train = pd.merge(df_train, df_center_info,
                    how="left",
                    left_on='center_id',
                    right_on='center_id')

df_train = pd.merge(df_train, df_meal_info,
                    how='left',
                    left_on='meal_id',
                    right_on='meal_id')
# Merge the test data with the branch and meal information.
df_test = pd.merge(df_test, df_center_info,
                   how="left",
                   left_on='center_id',
                   right_on='center_id')

df_test = pd.merge(df_test, df_meal_info,
                   how='left',
                   left_on='meal_id',
                   right_on='meal_id')

#Exploring dataset
df_train.describe()
df_test.describe()
df_train.isnull().sum()
df_test.isnull().sum()

#Feature Engineering
# Convert 'city_code' and 'region_code' into a single feature - 'city_region'.
df_train['city'] = \
        df_train['city_code'].astype('str') + '_' + \
        df_train['region_code'].astype('str')

df_test['city'] = \
        df_test['city_code'].astype('str') + '_' + \
        df_test['region_code'].astype('str')

# Feature Engineering
df_train['Month'] = df_train['week'].apply(lambda x: int(x / 4.5))
df_train['Year'] = df_train['week'].apply(lambda x: int(x / 52.143))
df_train['discount'] = df_train['base_price'] - df_train['checkout_price']
df_train['discountP'] = (df_train['base_price'] - df_train['checkout_price']) / df_train['base_price']

df_test['Month'] = df_train['week'].apply(lambda x: int(x / 4.5))
df_test['Year'] = df_train['week'].apply(lambda x: int(x / 52.143))
df_test['discount'] = df_test['base_price'] - df_test['checkout_price']
df_test['discountP'] = (df_test['base_price'] - df_test['checkout_price']) / df_test['base_price']

df_train['num_orders_log1p'] = np.log1p(df_train['num_orders'])

#Checking for outliers
sns.boxplot(df_train['num_orders'])
print(df_train[df_train.num_orders == df_train.num_orders.max()])
df_train.at[14050, 'num_orders'] = 136
sns.boxplot(df_train.num_orders)

#Creating dummies from categorical value
df_train = pd.get_dummies(df_train)
df_train.head()
list(df_train.columns)
df_test = pd.get_dummies(df_test)
df_test.head()
list(df_test.columns)

#Splitting dataset into training and testing set
X = df_train.drop(['num_orders','num_orders_log1p'], axis=1)
y = df_train['num_orders_log1p']
X.head()
y.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#Decission Tree
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_D = sc.fit_transform(X_train)
X_test_D = sc.transform(X_test)
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X_train_D, y_train)
y_pred = regressor.predict(X_test_D)
from sklearn.metrics import mean_squared_error
from math import sqrt

rms = sqrt(mean_squared_error(y_test, y_pred))
rms

#Linear Regression
# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train_D, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test_D)

# Predicting the Test set results
y_pred = regressor.predict(X_test_D)

#Random Forest
# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 50, random_state = 0)
regressor.fit(X_train_D, y_train)

# Predicting a new result
y_pred = regressor.predict(X_test_D)
rms = sqrt(mean_squared_error(y_test, y_pred))
rms

#Artificial Neural Network
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
def build_regressor():
    regressor = Sequential()
    regressor.add(Dense(units=66,kernel_initializer='normal',
                     activation='relu',input_dim=87))
    regressor.add(Dense(units=66,kernel_initializer='normal',
                     activation='relu'))
    regressor.add(Dense(units=66,kernel_initializer='normal',
                     activation='relu'))
    regressor.add(Dense(units=66,kernel_initializer='normal',
                     activation='relu'))
    regressor.add(Dense(units=66,kernel_initializer='normal',
                     activation='relu'))
    regressor.add(Dense(units=1,kernel_initializer='normal',
                     activation='relu'))
    regressor.compile(optimizer='sgd',loss='mean_squared_error')
    return regressor
regressor = KerasRegressor(build_fn= build_regressor,batch_size=10,epochs=100)
regressor.fit(X_train,y_train)
preds = regressor.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
rmse

#XGBoost
import xgboost as xgb
data_dmatrix = xgb.DMatrix(data=X,label=y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 1500)
xg_reg.fit(X_train,y_train)

preds = xg_reg.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
rmse

#Random search with XGBoost
import time
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
reg = xgb.XGBRegressor()
param_grid = {
        'silent': [False],
        'max_depth': [6, 10, 15, 20],
        'learning_rate': [0.001, 0.01, 0.1, 0.2, 0,3],
        'subsample': [0.5, 0.7, 0.8, 1.0],
        'colsample_bytree': [ 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bylevel': [0.4, 0.5,  0.7, 0.9, 1.0],
        'min_child_weight': [0.5, 1.0, 3.0, 5.0, 7.0, 10.0],
        'gamma': [0, 0.25, 0.5, 1.0],
        'reg_lambda': [0.1, 1.0, 5.0, 10.0, 50.0, 100.0],
        'n_estimators': [100]}
rs_reg = RandomizedSearchCV(reg, param_grid, n_iter=20,
                            n_jobs=1, verbose=2, cv=3,
                            refit=False, random_state=42)
print("Randomized search..")
search_time_start = time.time()
rs_reg.fit(X_train, y_train)
print("Randomized search time:", time.time() - search_time_start)
best_score = rs_reg.best_score_
best_params = rs_reg.best_params_
print("Best score: {}".format(best_score))
print("Best params: ")
for param_name in sorted(best_params.keys()):
    print('%s: %r' % (param_name, best_params[param_name]))
preds = xg_reg.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
rmse
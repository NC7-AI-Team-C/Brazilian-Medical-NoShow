import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
from sklearn.utils import all_estimators
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score

import tensorflow as tf
tf.random.set_seed(77)  # weight 난수값

import warnings
warnings.filterwarnings('ignore')

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=64
)

### kfold ###
n_splits = 21
random_state = 64
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성
from catboost import CatBoostRegressor
model = CatBoostRegressor()

#3. 훈련
model.fit(x_train, y_train)

#4. 출력(평가, 예측)
score = cross_val_score(model, x_train, y_train, cv=kfold)   # cv='cross validation'
print('cv r2score : ', score)

y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)
print('cv pred : ', y_predict)

y_predict = model.predict(x_test)
r2_score = r2_score(y_test, y_predict)
print('r2 스코어 : ', r2_score)

# #### feature importances ####
# print(model, " : ", model.feature_importances_)

# import matplotlib.pyplot as plt
# n_features = datasets.data.shape[1]
# plt.barh(range(n_features), model.feature_importances_, align='center')
# plt.yticks(np.arange(n_features), datasets.feature_names)
# plt.title('california Feature Importance')
# plt.ylabel('Feature')
# plt.xlabel('Importance')
# plt.ylim(-1, n_features)
# plt.show()
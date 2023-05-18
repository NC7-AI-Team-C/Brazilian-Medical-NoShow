import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, KFold, cross_val_score,cross_val_predict
from sklearn.utils import all_estimators
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

import tensorflow as tf
tf.random.set_seed(77)  # weight 난수값

import warnings
warnings.filterwarnings('ignore')

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.6, random_state=128,shuffle=True
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
model = RandomForestRegressor()

#3. 훈련
model.fit(x_train, y_train)

#4. 출력(평가, 예측)
score = cross_val_score(model, x_train, y_train, cv=kfold)

y_predict = cross_val_predict(x_test, y_test, cv=kfold)

result = model.score(x_test, y_test)
print('r2_score : ', result)

#### feature importances ####
print(model, " : ", model.feature_importances_)

import matplotlib.pyplot as plt
n_features = datasets.data.shape[1]
plt.barh(range(n_features), model.feature_importances_, align='center')
plt.yticks(np.arange(n_features), datasets.feature_names)
plt.title('california Feature Importance')
plt.ylabel('Feature')
plt.xlabel('Importance')
plt.ylim(-1, n_features)
plt.show()
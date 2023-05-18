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

### scaler ###
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성
from xgboost import XGBRegressor
model = XGBRegressor(random_state=123, n_estimators=1000, 
    learning_rate = 0.1, max_depth = 6, gamma= 1)

#3. 훈련
model.fit(x_train, y_train,
          early_stopping_rounds=20,
          eval_set = [(x_train, y_train), (x_test, y_test)],
          eval_metric='rmse')

        # eval_metric 회귀모델 : rmse, mae, rmsle ..
        # eval_metric 이진분류 : error, auc, logloss ..
        # eval_metric 다중분류 : merror, mlogloss ..

#4. 출력(평가, 예측)
score = cross_val_score(model, x_train, y_train, cv=kfold)   # cv='cross validation'
print('cv r2score : ', score)

y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)
print('cv pred : ', y_predict)

y_predict = model.predict(x_test)
r2_score = r2_score(y_test, y_predict)
print('r2 스코어 : ', r2_score)

# cv r2score :  [0.82484308 0.83558851 0.82130434 0.80921021 0.83075548 0.83778635
#  0.82056493 0.8243662  0.85033823 0.83761268 0.83081699 0.81516315
#  0.8422928  0.84580966 0.85956621 0.82931706 0.82641478 0.86139868
#  0.84546668 0.83662207 0.80644409]
# cv pred :  [1.3538415 2.5306191 2.0214303 ... 0.8525855 1.7637832 5.022213 ]
# r2 스코어 :  0.8414291131081154

# selectFromModel
from sklearn.feature_selection import SelectFromModel
thresholds = model.feature_importances_

for thresh in thresholds:
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    print(select_x_train.shape, select_x_test.shape)

    selection_model = XGBRegressor()
    selection_model.fit(select_x_train, y_train)
    y_predict = selection_model.predict(select_x_test)
    score = r2_score(y_test, y_predict)
    print("Thresh=%.3f, n=%d, R2:%.2f%%"
        %(thresh, select_x_train.shape[1], score*100))

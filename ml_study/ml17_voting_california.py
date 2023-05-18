import numpy as np
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import r2_score
from xgboost import XGBRFRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, shuffle=True, random_state=38
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
xgb = XGBRFRegressor(n_estimators = 300, n_jobs = -1)
lgbm = LGBMRegressor(n_estimators = 300, n_jobs = -1)
cat = CatBoostRegressor(n_estimators = 300)

model = VotingRegressor(
    estimators=[('xgb', xgb), ('lgbm', lgbm), ('cat', cat)],
    n_jobs=-1
)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
# y_predict = model.predict(x_test)
# score = accuracy_score(y_test, y_predict)
# print('voting 결과 : ', score)

regressors = [cat, lgbm, xgb]
for model in regressors:
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    score = r2_score(y_test, y_predict)
    class_names = model.__class__.__name__
    print('{0} 정확도 : {1: .4f}'.format(class_names, score))

## voting
# CatBoostRegressor 정확도 :  0.8501
# LGBMRegressor 정확도 :  0.8353
# XGBRFRegressor 정확도 :  0.6994

## 파라미트 넣고
# CatBoostRegressor 정확도 :  0.8441
# LGBMRegressor 정확도 :  0.8481
# XGBRFRegressor 정확도 :  0.6986

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.utils import all_estimators
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
import tensorflow as tf
tf.random.set_seed(77)
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=64
)

### kfold ###
n_splits = 5
random_state = 64
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

param = [
    {'n_estimators' : [100, 200], 'max_depth' : [6, 8, 10],
     'n_jobs' : [-1, 1, 2]},
    {'n_estimators' : [100, 500], 'min_samples_leaf' : [3, 5, 7]}

]

#2. 모델구성
from sklearn.model_selection import GridSearchCV
model = RandomForestClassifier(max_depth=6, min_samples_split=10)

#3. 훈련
import time
start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time() - start_time

print('걸린 시간 : ', end_time, '초')

##1.
# 최적의 파라미터 :  {'min_samples_leaf': 5, 'n_estimators': 100}   => 최적의 파라미터도 찾고 model에 넣어준다
# 최적의 매개변수 :  RandomForestClassifier(min_samples_leaf=5) => 이걸 찾기 위해서 그리드서치를 함
# 베스트 스코어 :  0.9583333333333334
# 모델 스코어 :  0.9666666666666667
# 걸린 시간 :  6.060913324356079 초


#4. 출력(평가, 예측)

result = model.score(x_test, y_test)
print('acc : ', result)

score = cross_val_score(model, x_train, y_train, cv=kfold)   # cv='cross validation'
print('cv acc : ', score)

y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)
print('cv pred : ', y_predict)

acc = accuracy_score(y_test, y_predict)
print('cv pred acc : ', acc)

# 걸린 시간 :  0.10994124412536621 초
# acc :  0.9666666666666667
# cv acc :  [1.         0.91666667 0.91666667 0.95833333 1.        ]
# cv pred :  [1 1 1 0 0 0 0 1 2 2 2 1 0 0 1 2 0 1 1 2 0 0 1 0 2 1 2 2 1 2]
# cv pred acc :  0.9
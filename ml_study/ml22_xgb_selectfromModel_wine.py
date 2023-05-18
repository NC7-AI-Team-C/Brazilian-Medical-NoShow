import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
from sklearn.utils import all_estimators
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import tensorflow as tf
tf.random.set_seed(77)  # weight 난수값

import warnings
warnings.filterwarnings('ignore')

#1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets.target
feature_name = datasets.feature_names
print(feature_name)

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
from xgboost import XGBClassifier
model = XGBClassifier()

#3. 훈련
model.fit(x_train, y_train, early_stopping_rounds=100,
          eval_set = [(x_train, y_train), (x_test, y_test)],
          eval_metric='merror')

#4. 출력(평가, 예측)
result = model.score(x_test, y_test)
print('acc : ', result)

score = cross_val_score(model, x_train, y_train, cv=kfold)   # cv='cross validation'
print('cv acc : ', score)

y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)
print('cv pred : ', y_predict)

acc = accuracy_score(y_test, y_predict)
print('cv pred acc : ', acc)

# acc :  0.9722222222222222

# cv acc :  [1.         1.         1.         1.         1.         0.85714286
#  1.         1.         0.85714286 0.85714286 1.         1.
#  1.         1.         0.85714286 0.85714286 1.         0.83333333
#  1.         1.         1.        ]
# cv pred :  [2 1 0 2 0 1 2 2 2 2 1 1 2 0 2 1 0 1 2 0 1 2 1 2 1 0 2 1 1 0 1 0 2 2 2 1]
# cv pred acc :  0.9166666666666666

from sklearn.feature_selection import SelectFromModel
thresholds = model.feature_importances_

print("=========== SelectFromModel ===============")
for thresh in thresholds:
    selection = SelectFromModel(model, threshold=thresh, prefit=True) 
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    print(select_x_train.shape, select_x_test.shape)
    selection_model = XGBClassifier(n_jobs=-1, 
    random_state=123, 
    n_estimators=1000, 
    learning_rate = 0.1,
    max_depth = 6, 
    gamma= 1,)
selection_model.fit(select_x_train, y_train)
y_predict = selection_model.predict(select_x_test)
score = accuracy_score(y_test, y_predict)
print("Thresh=%.3f, n=%d, Acc:%.2f%%"
        %(thresh, select_x_train.shape[1], score*100))

# 컬럼명 출력
selected_feature_indices = selection.get_support(indices=True)
selected_feature_names = [feature_name[i] for i in selected_feature_indices]
print(selected_feature_names)

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold
from sklearn.utils import all_estimators
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.filterwarnings('ignore')

#1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target

### kfold ###
n_splits = 21
random_state = 64
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)

#2. 모델구성
model = RandomForestClassifier()

#3. 훈련
model.fit(x, y)

#4. 출력(평가, 예측)
result = model.score(x, y)
print('acc : ', result)

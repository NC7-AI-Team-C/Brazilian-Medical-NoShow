import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.datasets import load_wine
from sklearn.preprocessing import MinMaxScaler
import time

#1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, shuffle=True, random_state=38
)

# scaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_split = 5
kfold = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=38)

parm = {
    'n_estimators' : [100],
    'random_state' : [38, 62, 72],
    'max_features' : [3, 4, 7]
}

#2. 모델 (Bagging)
bagging = BaggingClassifier(DecisionTreeClassifier(),
                            n_estimators=100,
                            n_jobs=-1,
                            random_state=38)

model = GridSearchCV(bagging, parm, cv=kfold, refit=True, n_jobs=-1)

#3. 훈련
start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time() - start_time

#4. 평가, 예측
result = model.score(x_test, y_test)

print('최적의 매개변수 : ', model.best_estimator_)
print('최적의 파라미터 : ', model.best_params_)
print('걸린시간 : ', end_time)
print('Bagging 결과 : ', result)

# 걸린시간 :  1.6732745170593262
# Bagging 결과 :  0.9722222222222222

# 최적의 매개변수 :  BaggingClassifier(estimator=DecisionTreeClassifier(), max_features=3,
#                   n_estimators=100, n_jobs=-1, random_state=62)
# 최적의 파라미터 :  {'max_features': 3, 'n_estimators': 100, 'random_state': 62}
# 걸린시간 :  2.546936511993408
# Bagging 결과 :  0.9722222222222222

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.utils import all_estimators
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

import tensorflow as tf
tf.random.set_seed(77)
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
path = '../AI_study/'
datasets = pd.read_csv(path + 'medical_noshow.csv')

print('columns : \n',datasets.columns)
print('head : \n',datasets.head(7))

x = datasets[['PatientId', 'AppointmentID', 'Gender', 'ScheduledDay',
       'AppointmentDay', 'Age', 'Neighbourhood', 'Scholarship', 'Hipertension',
       'Diabetes', 'Alcoholism', 'Handcap', 'SMS_received']]
y = datasets[['No-show']]
print(x.shape, y.shape)

print(x.info())  # info() 컬럼명, null값, data타입 확인

print(x.describe())

# 결과에 영향을 주지 않는 값 삭제
x = x.drop(['PatientId', 'AppointmentID'], axis=1)

print(x.shape)

x = x.fillna(np.NaN)

# 문자를 숫자로 변경
from sklearn.preprocessing import LabelEncoder
ob_col = list(x.dtypes[x.dtypes=='object'].index)    # 오브젝트 컬럼 리스트 추출
print(ob_col)
for col in ob_col:
    x[col] = LabelEncoder().fit_transform(x[col].values)
y = LabelEncoder().fit_transform(y.values)
# no = 0 , yes = 1

x = x.fillna(np.NaN)

print('columns : \n',x.columns)
print('head : \n',x.head(7))
print('y : ',y[0:8])

x_train, x_test, y_train, y_test = train_test_split(
x, y, test_size = 0.2, shuffle=True, random_state=62)

### kfold ###
n_splits = 5
random_state = 64
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

### scaler ###
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성
from sklearn.model_selection import GridSearchCV
model = LGBMClassifier(learning_rate=0.006478366591064283, max_depth=10,
                       num_leaves=29, feature_fraction=0.7057579511387295,
                        bagging_fraction=0.6653693411687419,
                        min_data_in_leaf=96, lambda_l1=6.606737188014566,
                        lambda_l2=9.596559651974772, n_estimators=370)

#3. 훈련
import time
start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time() - start_time

print('걸린 시간 : ', end_time, '초')


#4. 출력(평가, 예측)

result = model.score(x_test, y_test)
print('acc : ', result)

score = cross_val_score(model, x_train, y_train, cv=kfold)   # cv='cross validation'
print('cv acc : ', score)

y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)
print('cv pred : ', y_predict)

acc = accuracy_score(y_test, y_predict)
print('cv pred acc : ', acc)

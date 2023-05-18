import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.preprocessing import QuantileTransformer, PowerTransformer
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.filterwarnings('ignore')
import time


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
    x, y, shuffle=True, random_state=72, train_size=0.8
)

# 스케일링
sts = StandardScaler() 
mms = MinMaxScaler()
mas = MaxAbsScaler()
rbs = RobustScaler()
qtf = QuantileTransformer()                     # QuantileTransformer 는 지정된 분위수에 맞게 데이터를 변환함. 
                                                # 기본 분위수는 1,000개이며, n_quantiles 매개변수에서 변경할 수 있음
ptf1 = PowerTransformer(method='yeo-johnson')   # 'yeo-johnson', 양수 및 음수 값으로 작동
ptf2 = PowerTransformer(method='box-cox')       # 'box-cox', 양수 값에서만 작동

scalers = [sts, mms, mas, rbs, qtf, ptf1, ptf2]
for scaler in scalers:
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    model = RandomForestClassifier()
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    result = accuracy_score(y_test, y_predict)
    scale_name = scaler.__class__.__name__
    print('{0} 결과 : {1:.4f}'.format(scale_name, result), )


# [I 2023-05-17 14:56:03,950] Trial 4 finished with value: 0.6010585361440333 and parameters: {'n_estimators': 1367, 'depth': 13, 'fold_permutation_block': 151, 'learning_rate': 0.5053385423590736, 'od_pval': 0.960062127423458, 'l2_leaf_reg': 2.839749271495077, 'random_state': 877}. Best is trial 3 with value: 0.783814349045508.
# Best trial : score 0.783814349045508, /nparams {'n_estimators': 2699, 'depth': 10, 'fold_permutation_block': 196, 'learning_rate': 0.44717988001157216, 'od_pval': 0.5614356413869938, 'l2_leaf_reg': 0.38510957498206944, 'random_state': 1472}


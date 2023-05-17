import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from sklearn.covariance import EllipticEnvelope

import warnings
warnings.filterwarnings('ignore')

# 1. Data preprocessing #

path = './medical_noshow.csv'
medical_noshow = pd.read_csv(path)

# print(medical_noshow.columns)
# print(medical_noshow.head(10))

medical_noshow.AppointmentDay = pd.to_datetime(medical_noshow.AppointmentDay).dt.date
medical_noshow.ScheduledDay = pd.to_datetime(medical_noshow.ScheduledDay).dt.date
medical_noshow['PeriodBetween'] = medical_noshow.AppointmentDay - medical_noshow.ScheduledDay
# convert derived datetime to int
medical_noshow['PeriodBetween'] = medical_noshow['PeriodBetween'].dt.days

medical_noshow['diseaseCount'] = medical_noshow.Hipertension + medical_noshow.Diabetes + medical_noshow.Alcoholism
# print(datasets.PeriodBetween.describe())
x = medical_noshow[['PatientId', 'AppointmentID', 'Gender',	'ScheduledDay', 
              'AppointmentDay', 'PeriodBetween', 'Age', 'Neighbourhood', 
              'Scholarship', 'diseaseCount', 'Hipertension', 'Diabetes', 'Alcoholism', 
              'Handcap', 'SMS_received']]
y = medical_noshow[['No-show']]

# print(x.info())
# print(y.info())

## 1-1. correlation hit map ##

# sns.set(font_scale = 1)
# sns.set(rc = {'figure.figsize':(12, 8)})
# sns.heatmap(data = medical_noshow.corr(), square = True, annot = True, cbar = True)
# plt.show()

## 1-2. drop useless data ##

x = x.drop(['PatientId', 'AppointmentID','ScheduledDay', 'Hipertension', 'Diabetes', 'Alcoholism'], axis=1)
print(x.describe())
# print(x.shape)

outliers = EllipticEnvelope(contamination=.1)      
# 이상치 탐지 모델 생성
outliers.fit(x[['Age', '']])      
# 이상치 탐지 모델 훈련
predictions = outliers.predict(x[['Age']])       
# 이상치 판별 결과
outlier_indices = np.where(predictions == -1)[0]    
# 이상치로 판별된 행의 인덱스를 추출
x = x.drop(outlier_indices) 
# 데이터프레임에서 이상치 행을 삭제
y = y.drop(outlier_indices) 
# 데이터프레임에서 이상치 행을 삭제

print("이상치 정리 후",x.describe())
# print(x.shape, y.shape)

## 1-3. encoding object to int ##

encoder = LabelEncoder()

### char -> number
ob_col = list(x.dtypes[x.dtypes=='object'].index) ### data type이 object인 data들의 index를 ob_col 리스트에 저장
# print(ob_col)

for col in ob_col:
    x[col] = LabelEncoder().fit_transform(x[col].values)
y = LabelEncoder().fit_transform(y.values)

# print(x.describe())
# print('y:', y[0:8], '...')
print(x.info())

## 1-4. fill na data ##

x = x.fillna(np.NaN)
# # print(x.describe())

## 1-5. check dataset ##

print('head : \n',x.head(7))
print('y : ',y[0:7]) ### y : np.array

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.6, test_size=0.2, random_state=100, shuffle=True
)

scaler = PowerTransformer()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 3
random_state = 42
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

xgb = XGBClassifier(subsample = 1.0, reg_lambda = 0.5, reg_alpha = 0.1, n_estimators = 350, min_child_weight = 1, max_depth = 10, learning_rate = 0.075, gamma = 0.05, colsample_bytree = 1.0, colsample_bynode = 0.9, colsample_bylevel = 1,
                        # tree_method='gpu_hist', predictor='gpu_predictor', gpu_id=0   # gpu 사용시에만 입력하세요
                        ) 
# gridSearchCV로 산출한 best_parameter 적용
lgbm = LGBMClassifier(subsample = 0.6, reg_lambda = 0.9, reg_alpha = 0.2, num_leaves = 42, n_estimators = 500, min_data_in_leaf = 1, min_child_samples = 45, max_depth = 12, learning_rate = 0.1, feature_fraction = 0.85, colsample_bytree = 1.0,
                    #   device='gpu'   # gpu 사용시에만 입력하세요
                      ) 
# gridSearchCV로 산출한 best_parameter 적용
cat = CatBoostClassifier(subsample = 1.0, random_strength = 4, n_estimators = 700, learning_rate = 0.05, l2_leaf_reg = 6, depth = 9, colsample_bylevel = 0.65, border_count = 60, bagging_temperature = 0.12,
                        #  task_type='GPU',   # gpu 사용시에만 입력하세요
                         verbose=0) 
# gridSearchCV로 산출한 best_parameter 적용

model = VotingClassifier(
    estimators=[('xgb', xgb), ('lgbm', lgbm), ('cat', cat)],
    voting='hard',
    n_jobs=-1,
    verbose=0
)

model.fit(x_train, y_train)

y_voting_predict = model.predict(x_test)
voting_score = accuracy_score(y_test, y_voting_predict)

classifiers = [cat, xgb, lgbm]
for model in classifiers:
    model.fit(x_train, y_train, verbose=0)
    y_predict = model.predict(x_test)
    score = accuracy_score(y_test, y_predict)
    class_name = model.__class__.__name__
    print(class_name, "'s score : ", score)

print('voting result : ', voting_score)

# # 하이퍼 파라미터 적용
# CatBoostClassifier 's score :  0.7964353569166742
# XGBClassifier 's score :  0.7976115081878223
# LGBMClassifier 's score :  0.7981543472360445

# # 칼럼 전처리, 이상치 10% 삭제
# CatBoostClassifier 's score :  0.7976605276256844
# XGBClassifier 's score :  0.7921851667496267
# LGBMClassifier 's score :  0.7974614235938278
# voting result :  0.798456943753111

# # 칼럼 전처리, 이상치 0.1% 삭제
# CatBoostClassifier 's score :  0.7976454607199457
# XGBClassifier 's score :  0.7915327145121123
# LGBMClassifier 's score :  0.7965134706814581
# voting result :  0.7971926647045506

# # # 칼럼 전처리, 이상치 10% 삭제, n_splits = 8 # 변함없음 -> 롤백
# CatBoostClassifier 's score :  0.7976605276256844
# XGBClassifier 's score :  0.7921851667496267
# LGBMClassifier 's score :  0.7974614235938278
# voting result :  0.798456943753111

# # # 칼럼 전처리, 이상치 10% 삭제, n_splits = 8, MinMax -> std
# CatBoostClassifier 's score :  0.7976605276256844
# XGBClassifier 's score :  0.7922349427575909  # 0.0001상승
# LGBMClassifier 's score :  0.7974116475858636 # 0.00005하락
# voting result :  0.798456943753111

# # 질병 세개 통합
# CatBoostClassifier 's score :  0.7973618715778995
# XGBClassifier 's score :  0.7905425584868093
# LGBMClassifier 's score :  0.7970134395221503
# voting result :  0.799054255848681

# # 지역 칼럼 삭제 # acc 모두 하락 -> 롤백
# CatBoostClassifier 's score :  0.7948730711796914
# XGBClassifier 's score :  0.787954206072673
# LGBMClassifier 's score :  0.7936286709805873
# voting result :  0.7942757590841214

# # 투표방식 소프트로 전환 -> 롤백
# CatBoostClassifier 's score :  0.7973618715778995
# XGBClassifier 's score :  0.7905425584868093
# LGBMClassifier 's score :  0.7970134395221503
# voting result :  0.7981085116973619  # 하락

# 하이퍼 파라미터 랜덤서치 2차 적용
# CatBoostClassifier 's score :  0.7979591836734694
# XGBClassifier 's score :  0.7905923344947735
# LGBMClassifier 's score :  0.796316575410652
# voting result :  0.7977103036336486
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, PowerTransformer, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from sklearn.covariance import EllipticEnvelope

# 1. Data preprocessing #

path = 'C:/Users/bitcamp/Desktop/새 폴더/'
medical_noshow = pd.read_csv(path + 'medical_noshow.csv')

# print(medical_noshow.columns)
# print(medical_noshow.head(10))

medical_noshow.AppointmentDay = pd.to_datetime(medical_noshow.AppointmentDay).dt.date
medical_noshow.ScheduledDay = pd.to_datetime(medical_noshow.ScheduledDay).dt.date
medical_noshow['PeriodBetween'] = medical_noshow.AppointmentDay - medical_noshow.ScheduledDay
# convert derived datetime to int
medical_noshow['PeriodBetween'] = medical_noshow['PeriodBetween'].dt.days
# print(datasets.PeriodBetween.describe())
x = medical_noshow[['PatientId', 'AppointmentID', 'Gender',	'ScheduledDay', 
              'AppointmentDay', 'PeriodBetween', 'Age', 'Neighbourhood', 
              'Scholarship', 'Hipertension', 'Diabetes', 'Alcoholism', 
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

x = x.drop(['PatientId', 'AppointmentID','ScheduledDay'], axis=1)
# print(x.describe())
print(x.shape)
outliers = EllipticEnvelope(contamination=.10)      
# 이상치 탐지 모델 생성
outliers.fit(x[['Age']])      
# 이상치 탐지 모델 훈련
predictions = outliers.predict(x[['Age']])       
# 이상치 판별 결과
outlier_indices = np.where(predictions == -1)[0]    
# 이상치로 판별된 행의 인덱스를 추출
x = x.drop(outlier_indices) 
# 데이터프레임에서 이상치 행을 삭제
y = y.drop(outlier_indices) 
# 데이터프레임에서 이상치 행을 삭제
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
# print(x.info())

## 1-4. fill na data ##

x = x.fillna(np.NaN)
# # print(x.describe())

## 1-5. check dataset ##

# print('head : \n',x.head(7))
# print('y : ',y[0:7]) ### y : np.array

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.6, test_size=0.2, random_state=100, shuffle=True
)
#####스케일링#######
sts = StandardScaler() 
mms = MinMaxScaler()
mas = MaxAbsScaler()
rbs = RobustScaler()
qtf = QuantileTransformer() 

ptf1 = PowerTransformer(method='yeo-johnson')   # 'yeo-johnson', 양수 및 음수 값으로 작동
ptf2 = PowerTransformer(method='box-cox')          

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

# n_splits = 5
# kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# xgb = XGBClassifier(subsample = 1.0, reg_lambda = 0.5, reg_alpha = 0.1, n_estimators = 300, min_child_weight = 1, max_depth = 9, learning_rate = 0.1, gamma = 0.1, colsample_bytree = 1.0, colsample_bynode = 1, colsample_bylevel = 1) 
# # gridSearchCV로 산출한 best_parameter 적용
# lgbm = LGBMClassifier(subsample = 0.5, reg_lambda = 1, reg_alpha = 0.1, num_leaves = 31, n_estimators = 500, min_data_in_leaf = 1, min_child_samples = 50, max_depth = 9, learning_rate = 0.1, feature_fraction = 0.9, colsample_bytree = 1.0) 
# # gridSearchCV로 산출한 best_parameter 적용
# cat = CatBoostClassifier(subsample = 1.0, random_strength = 5, n_estimators = 500, learning_rate = 0.05, l2_leaf_reg = 7, depth = 9, colsample_bylevel = 0.5, border_count = 50, bagging_temperature = 0.1) 
# # gridSearchCV로 산출한 best_parameter 적용

# model = VotingClassifier(
#     estimators=[('xgb', xgb), ('lgbm', lgbm), ('cat', cat)],
#     voting='hard',
#     n_jobs=-1,
#     verbose=0
# )

# model.fit(x_train, y_train)

# y_voting_predict = model.predict(x_test)
# voting_score = accuracy_score(y_test, y_voting_predict)

# classifiers = [cat, xgb, lgbm]
# for model in classifiers:
#     model.fit(x_train, y_train)
#     y_predict = model.predict(x_test)
#     score = accuracy_score(y_test, y_predict)
#     class_name = model.__class__.__name__
#     print(class_name, "'s score : ", score)

# print('voting result : ', voting_score)

# # # 하이퍼 파라미터 적용
# # CatBoostClassifier 's score :  0.7964353569166742
# # XGBClassifier 's score :  0.7976115081878223
# # LGBMClassifier 's score :  0.7981543472360445

# # # 이상치 10% 삭제
# # CatBoostClassifier 's score :  0.7976605276256844
# # XGBClassifier 's score :  0.7921851667496267
# # LGBMClassifier 's score :  0.7974614235938278
# # voting result :  0.798456943753111



# from sklearn.feature_selection import SelectFromModel
# thresholds = model.feature_importances_
# print("=========== SelectFromModel ===============")
# for thresh in thresholds:
#     selection = SelectFromModel(model, threshold=thresh, prefit=True)
#     select_x_train = selection.transform(x_train)
#     select_x_test = selection.transform(x_test)
#     print(select_x_train.shape, select_x_test.shape)
    
#     selection_model = CatBoostClassifier()
#     selection_model.fit(select_x_train, y_train)
#     y_predict = selection_model.predict(select_x_test)
#     score = accuracy_score(y_test, y_predict)
#     print("Thresh=%.3f, n=%d, acc:%.2f%%"
#     %(thresh, select_x_train.shape[1], score*100))
    
#     # Thresh=249.000, n=6, acc:79.74%
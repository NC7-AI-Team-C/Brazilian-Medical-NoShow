import numpy as np
import pandas as pd
import time
import tensorflow as tf
tf.random.set_seed(62)
import warnings
warnings.filterwarnings('ignore')
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.metrics import accuracy_score
from sklearn.utils import all_estimators
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

##### 데이터 전처리 시작 #####
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

###상관계수 히트맵###
import matplotlib.pyplot as plt
import seaborn as sns

# pip install seaborn
sns.set(font_scale = 1.2)
sns.set(rc = {'figure.figsize':(12, 8)})   # 히트맵 데이터 맵 말고 히트맵 그림의 크기
sns.heatmap(data = x.corr(), #상관관계
            square = True,
            annot = True,
            cbar = True,
           )
##### 전처리 완료 #####

##### 훈련 구성 시작 #####
x_train, x_test, y_train, y_test = train_test_split(
x, y, test_size = 0.2, shuffle=True, random_state=62
)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

### kfold ###
n_splits = 35
random_state = 62
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

### scaler ###
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
xgb = XGBClassifier(n_estimators=951, learning_rate=0.7979977093999802,
                    random_state=1760)

lgbm = LGBMClassifier(learning_rate=0.006478366591064283, max_depth=10, num_leaves=29,
                      feature_fraction=0.7057579511387295, bagging_fraction=0.6653693411687419,
                      min_data_in_leaf=96, lambda_l1=6.606737188014566,
                      lambda_l2=9.596559651974772, n_estimators=370, random_state=1062)

cat = CatBoostClassifier(n_estimators=3744, depth=12, fold_permutation_block=52,
                         learning_rate=0.590550835248644, od_pval=0.4950728495185369,
                         l2_leaf_reg=1.5544576109556445, random_state=622)

model = VotingClassifier(
    estimators=[('xgb', xgb), ('lgbm', lgbm), ('cat', cat)],
    voting='hard',
    n_jobs=-1
)

# 정규화(Normalization)
x_train = x_train/255.0
x_test = x_test/255.0

#3. 컴파일, 훈련
model.fit(x_train, y_train)

classfiers = [cat, lgbm, xgb]
for model in classfiers:
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    score = accuracy_score(y_test, y_predict)
    class_names = model.__class__.__name__
    print('{0} 정확도 : {1: .4f}'.format(class_names, score))

#### feature importances ####
print(model, " : ", model.feature_importances_) # sequential model has no attribute feature_importances
n_features = x.shape[1]
plt.barh(range(n_features), model.feature_importances_, align='center')
plt.yticks(np.arange(n_features), ['Gender','ScheduledDay', 'AppointmentDay', 'Age', 'Neighbourhood', 'Scholarship', 'Hipertension', 'Diabetes', 'Alcoholism', 'Handcap', 'SMS_received'])
plt.title('noshow datset feature input importances')
plt.ylabel('feature')
plt.xlabel('importance')
plt.ylim(-1, n_features)
plt.show()

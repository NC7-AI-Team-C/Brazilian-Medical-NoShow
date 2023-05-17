import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# 1. Data preprocessing #

path = './medical_noshow.csv'
medical_noshow = pd.read_csv(path)

# print(medical_noshow.columns)
# print(medical_noshow.head(10))

x = medical_noshow[['PatientId', 'AppointmentID', 'Gender',	'ScheduledDay', 'AppointmentDay', 'Age', 'Neighbourhood', 'Scholarship', 'Hipertension', 'Diabetes', 'Alcoholism', 'Handcap', 'SMS_received']]
y = medical_noshow[['No-show']]

# print(x.info())
# print(y.info())

## 1-1. correlation hit map ##

sns.set(font_scale = 1)
sns.set(rc = {'figure.figsize':(12, 8)})
sns.heatmap(data = medical_noshow.corr(), square = True, annot = True, cbar = True)
# plt.show()

# ## 1-2. drop useless data ##

x = x.drop(['PatientId', 'AppointmentID'], axis=1)
# print(x.describe())

# ## 1-3. encoding object to int ##

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

# ## 1-4. fill na data ##

x = x.fillna(np.NaN)
# # print(x.describe())

# ## 1-5. check dataset ##

print('head : \n',x.head(7))
print('y : ',y[0:7]) ### y : np.array

# # 2. Modeling #

# ## 2-1. Dividing into training and test data ##

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.6, test_size=0.2, random_state=100, shuffle=True
)

# ## 2-2. create model ##

model = Sequential()

model.add(Dense(10, input_dim=11))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# print(model, " : ", model.feature_importances_) - sequential model has no attribute feature_importances
# n_features = medical_noshow.data.shape[1]
# plt.barh(range(n_features), model.feature_importances_, align='center')
# plt.yticks(np.arange(n_features), medical_noshow.feature_names)
# plt.title('iris datset feature input importances')
# plt.ylabel('feature')
# plt.xlabel('importance')
# plt.ylim(-1, n_features)
# plt.show()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

earlyStopping = EarlyStopping(
    monitor="val_loss",		
    # min_delta=0,		# 최소 성능 변화량 지정. 이 값보다 변화량의 절대값이 적다면 early stopping한다.
    patience=100,			# 지정값만큼의 epoch에서 변화량에 변화가 없다면 early stopping
    verbose=0,			  # 훈련 과정 보기/보지않기
    mode="auto",		  # min, max, auto모드가 있는데, min/max에서는 모니터링되는 변수의 양이 줄거나 느는 것이 멈추면 early stopping. auto에서는 모니터링되는 변수의 이름에서 자동 유추
    # baseline=None,	# 모니터링 되는 변수의 기준 값. 기준 값보다 개선되지 않으면 early stopping
    restore_best_weights=True,	# 모니터링된 변수의 최상의 값으로 epoch에서 모델 가중치를 복원할지 여부. 
)

mcp = ModelCheckpoint(
    filepath='./medical_noshow_mcp.hdf5',                     # hdf5파일을 저장할 위치 지정
    monitor="val_loss",    # 모니터링 할 변수 지정
    verbose=0,             # 훈련 과정 보이지 않게 하기/보이게 하기
    save_best_only=True, # True일 때 이미 저장한 체크포인트 파일이 더 뛰어나다면 업데이트 하지 않게 하기
    save_weights_only=False, # True일 때 가중치만 업데이트하기. False라면 파일 전체가 업데이트됨
    
    mode = "min", 
    # True인 경우 현재 저장 파일을 덮어쓸지 여부는 모니터링되는 양의 최대화 또는 최소화에 따라 결정됨. 
    # val_acc라면 max, val_loss라면 min이어야 함. auto는 모니터링 되는 변수가 fmeasure, acc라면 max로, 나머지는 min으로 설정됨. 
    
    # save_freq="epoch", # 설정한 epoch마다 저장됨.
    # options=None,# True : tf.train.CheckpointOptions / False : tf.saved_model.SaveOptions
    # initial_value_threshold=None, # 모델의 가중치가 지정값보다 나을 때만 파일을 업데이트함.
)

## 2-3. train model ##

start_time = time.time()

model.fit(
    x_train, y_train,
    epochs=1000,
    batch_size=128,
    validation_split=0.2,
    verbose=False
)

end_time = time.time() - start_time

loss, acc = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('acc : ', acc)
print('소요시간 : ', end_time)


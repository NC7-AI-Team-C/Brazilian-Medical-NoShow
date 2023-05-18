import numpy as np
import pandas as pd

data = pd.DataFrame([[2, np.nan, 6, 7, 10],
                     [2, 4, np.nan, 8, np.nan],
                     [2, 4, 6, 8, 10],
                     [np.nan, 4, np.nan, 8, np.nan]

])

data = data.transpose()
data.columns = ['x1', 'x2', 'x3', 'x4']

# print(data)
# print(data.shape)

# 결측치 확인
# print(data.isnull())
print(data.isnull().sum())
print(data.info())

#1. 결측치 삭제
# print(data.dropna(axis=1))
print(data.dropna(axis=1))
print(data.shape)
print(data)

#2. 특정값으로 대체
means = data.mean() # 평균
median = data.median()  # 중간값

data2 = data.fillna(means)
print(data2)

data3 = data.fillna(median)
print(data3)
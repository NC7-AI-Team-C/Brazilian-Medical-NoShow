import numpy as np
import pandas as pd

data = pd.DataFrame([[2, np.nan, 6, 7, 10],
                     [2, 4, np.nan, 8, np.nan],
                     [2, 4, 6, 8, 10],
                     [np.nan, 4, np.nan, 8, np.nan]

])

data = data.transpose()
data.columns = ['x1', 'x2', 'x3', 'x4']

# from sklearn.impute import IterativeImputer
# from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer

# imputer = SimpleImputer()   # 평균값으로 대체
# imputer = SimpleImputer(strategy='mean')    # 평균값
#imputer = SimpleImputer(strategy='median')  # 중간값
imputer = SimpleImputer(strategy='most_frequent')   # 가장 많이 사용되는 값
# imputer = SimpleImputer(strategy='constant', fill_value=777)    # 상수(default=0) 입력값
imputer.fit(data)
data2 = imputer.transform(data)
print(data2)
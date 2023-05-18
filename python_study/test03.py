import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])
print("Original :\n", a)                #\n은 줄바꿈

a_transpose = np.transpose(a)   # a 값이 순서대로 값이 나온다
print("Transpose :\n", a_transpose)
# Original :
#  [[1 2 3]
#  [4 5 6]]
# Transpose :
#  [[1 4]
#  [2 5]
#  [3 6]]

a_reshape = np.reshape(a, (3,2))    #reshape은 a 값이 변형해서 나열대로 설정해둔 행과열에 값이 나온다.
print("Reshape :\n", a_reshape)
# Reshape :
#  [[1 2]
#  [3 4]
#  [5 6]]
# 파이썬 기초교육
'''
# No. 4 리스트 숫자 출력
a = [37, 21, 75, 83, 10]
print(a)
print(a[0])

# 리스트 문자 출력
b = ['메이킷', '우진', '시은']
print(b)
print(b[0])
print(b[1])
print(b[2])

# 리스트 정수와 문자 출력
c = ['james', 26, 175.3, True]
print(c)

# 리스트 5번 문제
d = ['메이킷', '우진', '제임스', '시은']
# print(d[0], d[1])
# print(d[1], d[2], d[3])
# print(d[2], d[3])
# print(d)

print(d[0:2])
print(d[1:4])
print(d[2:4])
print(d[0:4])
'''

# No. 6 문제 ==> extend() 함수 사용하여 리스트 이어붙이기
a = ['우진', '시은']
b = ['메이킷', '소피아', '하워드']
a.extend(b)
print(a)
print(b)

c = ['나노', '누구']
d = ['메이킷', '소피아', '하워드']
d.extend(c)
print(c)
print(d)





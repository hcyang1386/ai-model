# 변수 선언, 변수 연산(덧셈)
a = 1
b = 2
c = a+b
print(a+b)

# 곱셈
print(a*b)

# 입력
a = int(input('first Number: '))
b = int(input('second Number: '))

# 처리
def adder(a, b):
    return a+b

#출력
print(adder(a , b))

print("=======================================")

# 자동화
while True:
    a = int(input('first Number: (999 for quit)'))
    if a == 999:
        print('bye~~')
        break
    else:
        b = int(input('second Number: '))
        print(adder(a,b))
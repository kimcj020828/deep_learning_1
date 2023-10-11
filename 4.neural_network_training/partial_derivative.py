'''
변수가 2개 이상의 여럿인 함수를 미분하는 방식을 편미분이라고 한다.
'''

'''
ex)
f(x0,x1) = x0**2 + x1**2라는 방정식이 있다고 하자.

case1) 이때 x0 = 3, x1 = 4 일때의 x0의 편미분 df/dx0의 값을 구하여라.
x0*x0 + 4**2

'''
from numerical_diff import numerical_diff

def f_tmp1(x0):
    return x0*x0 + 4.0**2    

print(numerical_diff(f_tmp1, 3.0))

'''
case2) 반면 x0 = 3이고, x1 = 4일때, x1의 편미분(x1을 제외한 모든 값을 상수 취급) df/x1 값을 구하면 다음과 같다
3**2 + x1*x1
'''
def f_tmp2(x1):
    return 3**2 + x1*x1

print(numerical_diff(f_tmp2,4))

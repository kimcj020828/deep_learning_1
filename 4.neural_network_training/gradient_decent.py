# eta 라는 기호는 갱신을 양을 나타낸다, 이를 신경망 학습에서는 학습률 learning rate라고 한다.
'''
한번의 학습으로 얼마나 학습해야할지, 즉 매개변수 값을 얼마나 갱신하느냐 정하는 과정이다.
학습률 값은 0.01이나 0.001 등 미리 특정 값으로 설정해두어야 하는데, 
일반적으로 이 값이 너무 크거나 작으면 '좋은 장소'를 찾아갈 수 없다.
신경망 학습에서는 보통 이 학습률 값을 변경하면서 올바르게 학습하고 있는지 확인하면서 진행한다.

# 학습률 같은 매개변수를 하이퍼파라미터(hyper parameter,초매개변수)라고 한다.
이는 가중치와 편향같은 신경망의 매개변수와는 성질이 다른 매개변수이다. 
신경망의 가중치 매개변수는 훈련데이터와 학습알고리즘에 의해 '자동'으로 획득되는 매개변수인 반면, 
학습률 같은 하이퍼파라미터는 사람이 직접 설정해야하는 매개변수인 것이다.
이 하이퍼 파라미터들은 여러 후보 값 중에서 시험을 통해 가장 잘 학습하는 값을 찾는 과정을 거쳐야 한다.

'''
from gradient import numerical_gradient
import numpy as np

# 경사하강법
def grident_descent(f, init_x,lr=0.01,step_num=100):
    # f: 최적화 하려는 함수
    # init_x: 초깃값
    # learning rate: 학습률
    # step_num: 경사법에 따른 반복 횟수
    
    x = init_x
    for i in range(step_num):
        grad = numerical_gradient(f,x)
        x -= lr*grad
    return x

# ex) f(x0,x1) = x0**2 + x1**2의 최소값을 구하시오
def function_2(x):
    return x[0]**2 + x[1]**2

init_x = np.array([-3.0,4.0])

# 예시1
print(grident_descent(function_2,init_x=init_x, lr = 0.1,step_num=100))

# 예시2: 학습률이 너무 큰 예:lr = 10.0
init_x = np.array([-3.0,4.0])
print(grident_descent(function_2,init_x=init_x, lr = 10.0,step_num=100))

# 예시3: 학습률이 너무 잒은 예:lr = 1e-10
init_x = np.array([-3.0,4.0])
print(grident_descent(function_2,init_x=init_x, lr = 1e-10,step_num=100))
'''
MNIST:손글씨 숫자 이미지 집합

28x28 크기의 이미지이고 레이블이 붙어있음
'''

import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist

# load하는데 몇 분 걸림
(x_train,t_train), (x_test,t_test) = load_mnist(flatten=True,normalize=False)

# 각 데이터의 형상 출력

print(x_train.shape) # (60000, 784) 
print(t_train.shape)#(60000,)
print(x_test.shape)#(10000, 784)
print(t_test.shape)#(10000,)
'''
기계학습 문제는 훈련데이터에 대한 손실함수의 값을 구하고, 그 값을 최대한 줄여주는 매개변수를 찾아냈다.

그러나 위 과정을 거치려면 모든 훈련데이터의 손실함수값을 구해야하는 막대한 자원이 필요하다.
그래서 무작위로 일부만 샘플링하여 근사치를 구하는 방식으로 적정자원으로 기대효과를 얻을 수 있는 방법을 취하려한다.
그 작업을 미니배치라 한다.
'''
import sys, os
sys.path.append(os.path.pardir)
import numpy as np
from dataset.mnist import load_mnist

(x_train,t_train), (x_test, t_test) = load_mnist(normalize=True,one_hot_label=True)

train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]
import sys, os
sys.path.append(os.path.pardir)
from dataset.mnist import load_mnist
import numpy as np
import pickle

def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax(a):
    c = np.max(a)
    exp_a=np.exp(a-c) #오버플로우 대책
    sum_exp_a=np.sum(exp_a)
    y=exp_a/sum_exp_a
    return y
    
    

def get_data():
    (x_train,t_train), (x_test,t_test) = load_mnist(normalize=True,flatten=True,one_hot_label=False)
    return x_test, t_test

def init_network():
    with open("sample_weight.pkl","rb") as f:
        network = pickle.load(f)
    return network

def predict(network, x):
    W1,W2,W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    
    a1 = np.dot(x,W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1,W2) + b2
    z2 = sigmoid(a2)
    z3 = np.dot(z2,W3) + b3
    y = sigmoid(z3)
    
    return y


x,t = get_data()
network = init_network()

accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network,x[i])
    p = np.argmax(y) # 확률이 제일 높은 원소의 인덱스를 얻는다.
    if p == t[i]:
        accuracy_cnt +=1

print("Accuracy:" + str(float(accuracy_cnt)/len(x)))

# 전처리 작업으로 이미지의 픽셀당 값을 0~255에서 0.0 ~ 1.0으로 정규화 함
'''
현업에서 전처리의 작업은 다양하다
ex)
- 데이터 전체 평균과 표준편차를 이용하여 데이터들이 0을 중심으로 분포하도록 이동하거나 확산범위를 제한하는 정규화 수행
- 그 외에도 전체 데이터를 균일하게 분포시키는 데이터 백색화(whitening)
'''
import sys,os
sys.path.append(os.path.pardir)
from neuralnet_mnist import get_data, init_network, predict
import numpy as np

x,t = get_data()
network = init_network()

batch_size = 100 # 배치크기
accuracy_cnt = 0

for i in range(0,len(x),batch_size): # range(start,end,step): start에서 end-1까지 정수를 step의 간격으로 차례차례 반환
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch,axis=1)
    accuracy_cnt += np.sum(p==t[i:i+batch_size])
    
print("Accuracy:" + str(float(accuracy_cnt)/len(x)))
    
   

'''
여기서 말하는 batch는 하나의 묶은 입력데이터를 가르킨다.
==> 배치처리의 이점은 컴퓨터로 개산할 때 큰 이점을 준다. 이미지 1장당 처리시간을 대폭줄여준다.
2가지 근거
- 수치 계산 라이브러리 대부분이 큰 배열을 효율적으로 처리할 수 있도록 고도로 최적화되어 있기 때문
- 커다란 신경망에서는 데이터 전송이 병목으로 작용하는 경우가 있으나, 배치 처리를 함으로써 버스에 주는 부하를 줄인다.
(느린 I/O를 통해 데이터를 읽은ㄴ 횟수가 줄어, 빠른 CPU나 GPU로 순수 계산을 수행한느 비율이 높아진다. )

'''
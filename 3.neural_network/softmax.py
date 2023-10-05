import numpy as np
a = np.array([0.3,2.9,4.0])

exp_a = np.exp(a) # 지수함수

print(exp_a)

sum_exp_a = np.sum(exp_a)
print(sum_exp_a)

y= exp_a/sum_exp_a
print(y)

# softmax의 자연상수의 지수값은 값이 워낙 커지기 때문에 오버플로우를 야기한다.
# 따라서 입력신호 중 최댓값을 빼주면 올바르게 계산이 가능하다
def softmax_with_overflow(a):
    exp_a=np.exp(a)
    sum_exp_a=np.sum(exp_a)
    y=exp_a/sum_exp_a
    return y

def softmax(a):
    c = np.max(a)
    exp_a=np.exp(a-c) #오버플로우 대책
    sum_exp_a=np.sum(exp_a)
    y=exp_a/sum_exp_a
    return y
    
    
a = np.array([0.3,2.9,4.0])
y = softmax(a)
print(y)
print(np.sum(y)) # 1

'''
softmax의 특징
- 출력의 총합은 1이다!!!!!!!! 
    => 확률로 해석할 수 있다.
        ==> 분류에서 몇번째 클래스의 확률이 가장 높을지 알 수 있다.
- 소프트맥스의 인자로 입력되는 값과 출력하는 값의 대소구분 관계는 입력전과 출력전이 같다/
    =>ex) 입력전 1번 인덱스가 가장 크다면, 소프트맥스를 통과해도 1번 인덱스가 제일 크다.
    =>지수함수의 계산으로 사용되는 자원낭비를 줄이고자 현업에서는 소프트맥스 함수를 생략하는 것이 일반적이다.
- 소프트맥스의 출력은 0~1.0 사이의 실수이다.

3.5.4 출력층의 뉴련수: 풀려는 문제에 맞게 설정
=> '분류'에서는 분류하고 싶은 클래스의 수로 설정 
'''

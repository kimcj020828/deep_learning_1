import numpy as np

# 원앤 핫 방식
# def cross_entropy_error(y,t):
#     if y.ndim == 1:
#         t = t.reshape(1, t.size)
#         y = y.reshape(1, y.size)
        
#     batch_size = y.shape[0]
#     return -np.sum(t*np.log(y+1e-7))/batch_size

# 데이터의 수치가 원앤 핫이 아닌, 직접 수치가 주어지는 경우
def cross_entropy_error(y,t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
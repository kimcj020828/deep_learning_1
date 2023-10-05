import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))

def identity_function(x):
    return x
    
X = np.array([1.0, 5.0])
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B1 = np.array([0.1,0.2,0.3])

print(W1.shape)
print(X.shape)
print(B1.shape)

A1 = np.dot(X,W1) + B1

Z1 = sigmoid(A1)

print(A1)
print(Z1)


W2 = np.array([[0.1,0.4], [0.2,0.5], [0.3,0.6]])
B2 = np.array([0.1,0.2])

print(Z1.shape)
print(W2.shape)
print(B2.shape)

A2 = np.dot(Z1,W2) + B2
Z2 = sigmoid(A2)

print (A2)
print (Z2)

W3 = np.array([[0.1,0.3], [0.2,0.4]])
B3 = np.array([0.1,0.2])

A3 = np.dot(Z2,W3) + B3
Y = identity_function(A3)

print(A3)
print(Y)

x = np.arange(0,np.size(Z1), 1) 
y = Z1
x2 = np.arange(0,np.size(Z2), 1) 
y2 = Z2
x3 = np.arange(0,np.size(Y), 1) 
y3 = Y

plt.plot(x,y)
plt.plot(x2,y2)
plt.plot(x3,Y)
plt.ylim(-0.1,1.1) # y축 범위지정
plt.show()
    
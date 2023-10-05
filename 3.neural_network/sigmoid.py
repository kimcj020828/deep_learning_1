import numpy as np
import matplotlib.pyplot as plt
def sigmoid(x):
    return 1/(1+np.exp(-x))


#x = np.array([-1.0,1.0,2.0])
x = np.arange(-5.0,5.0,0.1)


#print(sigmoid(x))
y = sigmoid(x)

plt.plot(x,y)
plt.ylim(-0.1,1.1)
plt.show()
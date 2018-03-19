
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


# In[ ]:


traindf = pd.read_csv('train.csv', header=None)
traindf.head()


# In[ ]:


xdf = traindf.iloc[:,0:7]
xdf[7] = 1
x = xdf.as_matrix()


# In[ ]:


ydf = traindf.iloc[:,7]
y = ydf.as_matrix()


# In[ ]:


def gradient(x, y, w):
    grad = []
    for j in range(len(w)):
        s = 0
        for i in range(x.shape[0]):
            s += (y[i]-w.dot(x[i]))*x[i][j]
        grad.append(-s)
    return np.array(grad)


# In[ ]:


def costFunction(x, y, w):
    j = 0 
    for i in range(x.shape[0]):
        j += (y[i]-w.dot(x[i]))**2
    return j / 2


# In[ ]:


def batchGradientDescent(x, y, r=0.1):
    w = np.zeros(x.shape[1])
    j = costFunction(x, y, w)
    js = []
    js.append(j)
    converge = False
    t = 0
    while t < 10000:
        w2 = w - r * gradient(x, y, w)
        j2 = costFunction(x, y, w2)
        if j2 > j:
            return False, w, j, js
        js.append(j2)
        if np.linalg.norm(w2-w) < 1e-6:
            return True, w2, j2, js
        w = w2
        j = j2
        t += 1
    return False, w, j, js


# In[ ]:


r = [1, 0.6, 0.3, 0.1, 0.06, 0.03, 0.01, 0.006, 0.003, 0.001]


# In[ ]:


for gamma in r:
    isConverged, weight, cost, costs = batchGradientDescent(x, y, gamma)
    if isConverged:
        print('learning rate: {}'.format(gamma))
        print('weight: {}'.format(weight))
        print('cost on training data: {}'.format(cost))
        break


# In[ ]:


np.linalg.inv(x.transpose().dot(x)).dot(x.transpose()).dot(y)


# In[ ]:


testdf = pd.read_csv('test.csv', header=None)
testdf.head()


# In[ ]:


xdf2 = testdf.iloc[:,0:7]
xdf2[7] = 1
x2 = xdf2.as_matrix()


# In[ ]:


ydf2 = testdf.iloc[:,7]
y2 = ydf2.as_matrix()


# In[ ]:


print('cost on testing data: {}'.format(costFunction(x2,y2,weight)))


# In[ ]:


plt.figure(figsize=(15,5))
plt.plot(costs)
# plt.xlim([0,10])
plt.xlabel('step')
plt.ylabel('cost')
plt.show()


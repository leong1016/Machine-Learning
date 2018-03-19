
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


def costFunction(x, y, w):
    j = 0 
    for i in range(x.shape[0]):
        j += (y[i]-w.dot(x[i]))**2
    return j / 2


# In[ ]:


def stochasticGradientDescent(x, y, r=0.1):
    w = np.zeros(x.shape[1])
    js = []
    js.append(costFunction(x, y, w))
    t = 0 
    while t < 10000:
        i = np.random.randint(x.shape[0])
        grad = []
        for j in range(len(w)):
            grad.append((y[i]-w.dot(x[i]))*x[i][j])
        grad = np.array(grad)
        w = w + r * grad
        cost = costFunction(x, y, w)
        js.append(cost)
        t = t + 1
    return w, cost, js


# In[ ]:


weight, cost, costs = stochasticGradientDescent(x, y, 0.001)


# In[ ]:


print('learning rate: {}'.format(0.001))
print('weight: {}'.format(weight))
print('cost on training data: {}'.format(cost))


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


plt.figure(figsize=[15,5])
plt.plot(costs)
# plt.xlim([00, 53000])
plt.xlabel('step')
plt.ylabel('cost')
plt.show()


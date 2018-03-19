
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


traindf = pd.read_csv('train.csv', header=None)
traindf.insert(4,'b',1)
traindf.head()


# In[ ]:


trainmtx = traindf.as_matrix()


# In[ ]:


trainmtx


# In[ ]:


def Averagedperceptron(trainmtx, r=0.1, t=10):
    n = trainmtx.shape[0]
    d = trainmtx.shape[1] - 1
    w = np.zeros(d)
    a = np.zeros(d)

    tt = 0
    while tt < t:
        np.random.shuffle(trainmtx)
        x = trainmtx[:,:-1]
        y = (trainmtx[:,-1] - 1/2) * 2
        for i in range(n):
            if w.dot(x[i]) * y[i] <= 0:
                w = w + r * y[i] * x[i]
            a = a + w
        tt += 1
    return a


# In[ ]:


weight = Averagedperceptron(trainmtx)
print('weight vector: {}'.format(weight))


# In[ ]:


testdf = pd.read_csv('test.csv', header=None)
testdf.head()


# In[ ]:


xdf = testdf.iloc[:,0:4]
xdf[4] = 1
x = xdf.as_matrix()


# In[ ]:


ydf = testdf.iloc[:,4]
ydf = (ydf-1/2)*2
y = ydf.as_matrix()


# In[ ]:


pred = x.dot(weight.transpose())*y
accuracy = (pred>=0).sum() / y.size
print('Accuracy: {0:.5f}'.format(accuracy))
print('Error: {0:.5f}'.format(1-accuracy))


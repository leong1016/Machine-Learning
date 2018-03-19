
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


def votedPerceptron(trainmtx, r=0.1, t=10):
    n = trainmtx.shape[0]
    d = trainmtx.shape[1] - 1
    w = [np.zeros(d)]
    c = [1]

    tt = 0
    while tt < t:
        np.random.shuffle(trainmtx)
        x = trainmtx[:,:-1]
        y = (trainmtx[:,-1] - 1/2) * 2
        for i in range(n):
            if w[-1].dot(x[i]) * y[i] <= 0:
                w.append(w[-1] + r * y[i] * x[i])
                c.append(1)
            else:
                c[-1] += 1
        tt += 1
    return np.array(w), np.array(c)


# In[ ]:


weight, count = votedPerceptron(trainmtx)


# In[ ]:


for i in range(count.size):
    rounded = ['%.5f' % e for e in weight[i]]
    string = '('+','.join(str(e) for e in rounded)+')'
    print('{} : {}'.format(string,count[i]))


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


predmtx = (x.dot(weight.transpose())>=0)-1/2
pred = predmtx.dot(count.transpose())


# In[ ]:


correct = ((pred * y)>=0).sum()
accuracy = correct / y.size
print('Accuracy: {0:.5f}'.format(accuracy))
print('Error: {0:.5f}'.format(1-accuracy))


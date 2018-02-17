
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


train = pd.read_csv('train.csv', header=None)
test = pd.read_csv('test.csv', header=None)


# In[3]:


data = train.iloc[:, :-1]
label = train.iloc[:, -1]


# In[4]:


values = {}
for col in data:
    values[col] = data[col].unique()


# In[5]:


class TestNode:
    
    def __init__(self, col, depth):
        self.type = 'test'
        self.col = col
        self.depth = depth
        self.branch = []
    
    def add_branch(self, branch):
        self.branch.append(branch)


# In[6]:


class Branch:
    
    def __init__(self, value, parent, child):
        self.value = value
        self.parent = parent
        self.child = child


# In[7]:


class LeafNode:
    
    def __init__(self, value, depth):
        self.type = 'leaf'
        self.value = value
        self.depth = depth


# In[8]:


def entropy(label):
    if (len(label.unique()) == 1):
        return 0
    total = len(label)
    count = label.value_counts().values
    pi = count/total
    logpi = np.log2(pi)
    h = -np.dot(pi, logpi)
    return h


# In[9]:


def majority_error(label):
    total = len(label)
    majority = label.value_counts().max()
    me = (total - majority) / total
    return me


# In[26]:


def information_gain(data, label, way):
    
    total = len(label)
    if way == 'entropy':
        h_prev = entropy(label)
    else:
        h_prev = majority_error(label)

    attr = data.name
    concated = pd.concat([data, label], axis=1)
    grouped = concated.groupby(attr)

    h_next = 0
    for name, group in grouped:
        count = len(group)
        p = count/total
        if way == 'entropy':
            h = entropy(group.iloc[:,-1])
        else:
            h = majority_error(group.iloc[:,-1])
        h_next += p * h

    ig = h_prev - h_next
    return ig


# In[27]:


def find_purest(data, label, way):

    best_ig = -1
    for col in data:
        ig = information_gain(data[col], label, way)
        if (ig > best_ig):
            best_ig = ig
            best_col = col
    return best_col


# In[28]:


def most_common(label):
    return label.value_counts().idxmax()


# In[29]:


def ID3(data, values, label, max_depth, depth=0, way='entropy'):
    
    #create leaf node if pure
    if entropy(label) == 0:
        return LeafNode(most_common(label), depth)
    
    #create leaf node if reach max_depth
    if depth == max_depth:
        return LeafNode(most_common(label), depth)
    
    #create leaf node if no atrributes to split
    if find_purest(data, label, way) == -1:
        return LeafNode(most_common(label), depth)
    
    best_col = find_purest(data, label, way)
    root = TestNode(best_col, depth)
    
    concated = pd.concat([data, label], axis=1)
    grouped = concated.groupby(best_col)
    
    for value in values[best_col]:

        if value in grouped.groups.keys():
            group = grouped.get_group(value)
            newgroup = group.drop(best_col, axis=1)
            newdata = newgroup.iloc[:,:-1]
            newlabel = newgroup.iloc[:,-1]
            temp = ID3(newdata, values, newlabel, max_depth, depth+1, way)
        else:
            temp = LeafNode(most_common(label), depth+1)
            
        branch = Branch(value, root, temp)
        root.add_branch(branch)
    
    return root


# In[30]:


def print_id3(tree):
    if (tree.type == 'test'):
        print('TEST: {}'.format(tree.col))
    elif (tree.type == 'leaf'):
        print('LEAF: {}'.format(tree.value))
        print('************')
        return;
    for i in tree.branch:
        print('VALUE: {}'.format(i.value))
        print_id3(i.child)


# In[31]:


def single_predict(root, row):
    while root.type != 'leaf':
        test = root.col
        value = row[test]
        branches = root.branch
        for branch in branches:
            if branch.value == value:
                root = branch.child
    return root.value


# In[32]:


def batch_predict(tree, test):
    predictions = []
    for index, row in test.iterrows():
        predictions.append(single_predict(tree, row))
    result = test
    result['pred'] = predictions
    return result


# In[36]:


def table_generator(way):
    print(way)
    for i in range(1, 8):
        if way == 'entropy':
            tree = ID3(data, values, label, i, way='entropy')
        else:
            tree = ID3(data, values, label, i, way='majority error')
        result1 = batch_predict(tree, train)
        compare1 = result1.iloc[:,-2] == result1.iloc[:,-1]
        accuracy1 = (len(compare1) - compare1.sum()) / len(compare1)
        result2 = batch_predict(tree, test)
        compare2 = result2.iloc[:,-2] == result2.iloc[:,-1]
        accuracy2 = (len(compare2) - compare2.sum()) / len(compare2)
        print('{0:d} & {1:.4f} & {2:.4f} \\\\'.format(i, accuracy1, accuracy2))


# In[37]:


table_generator('entropy')


# In[38]:


table_generator('majority_error')


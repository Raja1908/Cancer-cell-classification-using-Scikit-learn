#!/usr/bin/env python
# coding: utf-8

# In[13]:


import sklearn
from sklearn import datasets


# In[14]:


data=datasets.load_breast_cancer()
print(type(data))


# In[15]:


# Organize our data 
label_names = data["target_names"] 
labels = data['target'] 
feature_names = data['feature_names'] 
features = data['data'] 


# In[16]:


print(label_names)


# In[17]:


print(labels)


# In[18]:


print(feature_names)


# In[19]:


print(features)


# In[20]:


from sklearn.model_selection import train_test_split
train,test,train_labels,test_labels=train_test_split(features,labels,test_size=0.33,random_state=42)


# In[21]:


from sklearn.naive_bayes import GaussianNB
gnb=GaussianNB()
model=gnb.fit(train,train_labels)


# In[22]:


predictions=gnb.predict(test)
print(predictions)


# In[23]:


from sklearn.metrics import accuracy_score 
  
# evaluating the accuracy 
print(accuracy_score(test_labels, predictions)) 


# In[ ]:





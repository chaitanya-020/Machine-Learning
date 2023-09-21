#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from pgmpy.models import BayesianModel
from pgmpy.estimators import HillClimbSearch, MaximumLikelihoodEstimator
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination
from hmmlearn.hmm import GaussianHMM
from sklearn.metrics import classification_report

import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv('creditcard.csv')
df.head()


# In[3]:


df.drop('Time',inplace=True,axis=1)


# In[4]:


df


# In[5]:


correlation_matrix = df.corr()


# In[6]:


correlation_matrix


# In[7]:


fig = plt.subplots(figsize=(15,15)) 
sns.heatmap(correlation_matrix, vmax= 1 )


# In[8]:


df = df.sample(n=10000, random_state=42)


# In[9]:


df


# In[10]:


X_subset = df.drop('Class', axis=1)
y_subset = df['Class']


# In[11]:


X_subset


# In[12]:


variable_mapping = {i: col for i, col in enumerate(X_subset.columns)}
renamed_X_subset = X_subset.rename(columns=variable_mapping)


# In[13]:


X_subset.rename(columns={'Class': 'Class_node'}, inplace=True)


# In[14]:


X_subset=renamed_X_subset
X_subset


# In[15]:


col=list(df.columns)
col


# In[16]:


X_train,X_test,y_train,y_test=train_test_split(X_subset,y_subset,test_size=0.2,random_state=2)


# In[17]:


len(col)


# In[18]:


bbn_model = BayesianModel()
for i in range(len(col)-1):
    bbn_model.add_edge(col[i],col[i+1])


# In[19]:


bbn_model.nodes()


# In[20]:


bbn_model.fit(df, estimator=BayesianEstimator)


# In[ ]:


y_pred=bbn_model.predict(X_test)


# In[ ]:


acc=accuracy_score(y_test,y_pred)
print("accuracy=",acc*100)


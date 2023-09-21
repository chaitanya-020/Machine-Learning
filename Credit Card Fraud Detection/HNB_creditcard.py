#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !pip install imblearn


# In[2]:


# !pip install hmmlearn


# In[3]:


# pip install --upgrade scikit-learn


# In[4]:


# !pip install pgmpy==0.1.15


# In[33]:


import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,GridSearchCV
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


# In[34]:


df = pd.read_csv('creditcard.csv')
df.head()


# In[35]:


df.shape
df.astype(float)


# In[36]:


df.info()


# In[37]:


df.describe()


# In[38]:


classes = df['Class'].value_counts()
classes


# In[39]:


df.dtypes


# In[40]:


df=df.sample(n=100000,axis=0)


# In[41]:


sns.countplot(x=df['Class'])
df["Class"].value_counts()


# In[42]:


fraud = df[df["Class"] == 1]
non_fraud = df[df["Class"] == 0]


# In[43]:


fraud.describe()


# In[44]:


non_fraud.describe()


# In[45]:


# Adding titles to the plots and axes
plt.rcParams["figure.figsize"] = "8,6"
plt.title("Distribution of Amount over Both Classes")
plt.xlabel("Amount")
plt.ylabel("Class")

# Plotting the Amount column vs. Class Column
plt.scatter(df["Amount"],df["Class"])


# In[46]:


# Adding titles to the plots and axes
plt.title("Distribution of Time over Both Classes")
plt.xlabel("Time")
plt.ylabel("Class")

# Plotting the Time column vs. Class Column
plt.scatter(df["Time"],df["Class"])


# In[47]:


correlation = df.corr()
fig = plt.subplots(figsize=(15,15)) 
sns.heatmap(correlation, vmax= 1 )


# In[48]:


array = df.values
inf_indices = np.where(np.isinf(array))
nan_indices = np.where(np.isnan(array))
print(inf_indices, type(inf_indices))
print(nan_indices, type(nan_indices))
for row, col in zip(*inf_indices):
    array[row,col] = -1
    
for row, col in zip(*nan_indices):
    array[row,col] = 0
#array[]
X = array[:, 0:30]
y = array[:, 30]
ros = RandomOverSampler(random_state=42)
X_res, y_res = ros.fit_resample(X, y)

validation_size = 0.20
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res,
    test_size=validation_size, random_state=42)


# In[49]:


X


# In[50]:


X_train.shape


# In[51]:


y_train.shape


# In[52]:


sns.countplot(x=y)


# In[53]:


sns.countplot(x=y_train)


# # Hidden Naive Bayes

# In[74]:


param_grid = {'var_smoothing': [1e-19,1e-20]}

hidden_naive_bayes = GaussianNB()

grid_search = GridSearchCV(hidden_naive_bayes, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_model = grid_search.best_estimator_


# In[75]:


y_pred = best_model.predict(X_test)


# In[76]:


y_pred


# In[77]:


accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy*100)


#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE

from imblearn.over_sampling import RandomOverSampler 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from pgmpy.models import BayesianModel
from pgmpy.estimators import HillClimbSearch, MaximumLikelihoodEstimator
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination
from hmmlearn.hmm import GaussianHMM
from sklearn.metrics import classification_report

from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import RandomOverSampler 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix

import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv('creditcardcsvpresent.csv')
df


# In[3]:


df.shape


# In[4]:


df.info()


# In[5]:


df.head()


# In[6]:


dfFraud = df.loc[df.isFradulent == 'Y']
dfNonFraud = df.loc[df.isFradulent == 'N']


# In[7]:


dfFraud.shape


# In[8]:


dfNonFraud.shape


# In[9]:


df.describe()


# In[10]:


sns.countplot(x='isFradulent', data=df)
plt.title('Distribution of Target Variable')
plt.show()


# In[11]:


corr_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


# In[12]:


categorical_features = ['Is declined', 'isForeignTransaction', 'isHighRiskCountry', 'isFradulent']


# In[13]:


label_encoder = LabelEncoder()
for feature in categorical_features:
    df[feature] = label_encoder.fit_transform(df[feature])


# In[14]:


df


# In[15]:


X_hbn=df.drop('isFradulent',axis=1)
y_hbn=df['isFradulent']


# In[16]:


scaler = StandardScaler()
scaled_X = scaler.fit_transform(X_hbn)


# In[17]:


scaled_X


# In[18]:


scaled_X_df = pd.DataFrame(scaled_X, columns=X_hbn.columns)
X_hbn=scaled_X_df
X_hbn


# In[19]:


sns.countplot(x=y_hbn)


# In[20]:


X_train, X_test, y_train, y_test = train_test_split(X_hbn, y_hbn, test_size=0.2, random_state=42)


# In[21]:


oversampler = RandomOverSampler(random_state=21)
X_resampled, y_resampled = oversampler.fit_resample(X_train, y_train)


# In[22]:


sns.countplot(x=y_resampled)


# In[23]:



param_grid = {'var_smoothing': [1e+10,1e+11]}

hidden_naive_bayes = GaussianNB()

grid_search = GridSearchCV(hidden_naive_bayes, param_grid=param_grid, cv=5)
grid_search.fit(X_resampled, y_resampled)

best_params = grid_search.best_params_
best_model = grid_search.best_estimator_


# In[24]:


y_pred_hbn= best_model.predict(X_test)


# In[25]:


y_pred_hbn


# In[26]:


accuracy = accuracy_score(y_test, y_pred_hbn)
print("Accuracy:", accuracy)

precision = precision_score(y_test, y_pred_hbn)
print("Precision:", precision)

recall = recall_score(y_test, y_pred_hbn)
print("Recall:", recall)

f1 = f1_score(y_test, y_pred_hbn)
print("F1-score:", f1)


# In[27]:


cm = confusion_matrix(y_test, y_pred_hbn)
print("Confusion Matrix:")
print(cm)


# In[28]:


cm_df = pd.DataFrame(cm, index=['Actual 0', 'Actual 1'], columns=['Predicted 0', 'Predicted 1'])


# In[29]:


cm_df


# In[30]:


plt.figure(figsize=(8, 6))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')

plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.xticks([0.5, 1.5], ['0', '1'])
plt.yticks([0.5, 1.5], ['0', '1'])

plt.show()


# In[31]:


report = classification_report(y_test,y_pred_hbn)

print(report)


# # BAYESIAN BELIEF NETWORK

# In[32]:


from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from sklearn.metrics import accuracy_score, confusion_matrix


# In[33]:


from sklearn.base import BaseEstimator, ClassifierMixin
class BBNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.bbn_model = None

    def fit(self, X, y):
        self.bbn_model = BayesianModel()
        col = df.columns
        states = {col[i]: ['0', '1'] for i in range(len(col) - 1)} 
        self.bbn_model.add_nodes_from(states.keys())
        for i in range(len(col) - 1):
            self.bbn_model.add_edge(col[i], col[i + 1])
        self.bbn_model.fit(df, estimator=MaximumLikelihoodEstimator, complete_samples_only=False)
        return self

    def predict(self, X):
        return self.bbn_model.predict(X)


# In[34]:



col=df.columns
X=df.drop('isFradulent',axis=1)
y=df['isFradulent']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
oversampler = RandomOverSampler(random_state=21)
X_train_over, y_train_over = oversampler.fit_resample(X_train, y_train)


# In[35]:


bbn_classifier = BBNClassifier()
bbn_classifier.fit(X_train_over, y_train_over)

predictions = bbn_classifier.predict(X_test)


# In[47]:


accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)


precision = precision_score(y_test, predictions)
print("Precision:", precision)


recall = recall_score(y_test, predictions)
print("Recall:", recall)


f1 = f1_score(y_test, predictions)
print("F1-score:", f1)


# In[ ]:


cm = confusion_matrix(predictions,y_test)
print("Confusion Matrix:")
print(cm)


# In[45]:


cm_df = pd.DataFrame(cm, index=['Actual 0', 'Actual 1'], columns=['Predicted 0', 'Predicted 1'])


# In[41]:


plt.figure(figsize=(8, 6))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')

plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.xticks([0.5, 1.5], ['0', '1'])
plt.yticks([0.5, 1.5], ['0', '1'])

plt.show()


# In[42]:


report = classification_report(predictions,y_test)
print(report)


# In[ ]:





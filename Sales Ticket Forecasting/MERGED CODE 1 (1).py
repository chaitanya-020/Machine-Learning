#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from matplotlib import pyplot
import matplotlib.pyplot as plt
get_ipython().system('pip install pmdarima')
from pmdarima import auto_arima
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from statsmodels.tools.eval_measures import rmse
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
from sklearn.linear_model import LinearRegression


# In[2]:


data=pd.read_csv("Book1.csv")
data.rename(columns = {'sales':'ticket_count'}, inplace = True)
data


# In[3]:


warnings.filterwarnings("ignore")


# In[4]:


data['date'] = pd.to_datetime(data['date'])
data = data.set_index('date').asfreq('D')


# In[5]:


data =data.dropna()


# In[6]:


def arima_sarima(data):
    rmse_error=0
    result = seasonal_decompose(data['ticket_count'],model ='multiplicative',period=1)
    stepwise_fit = auto_arima(data['ticket_count'], start_p = 1, start_q = 1,max_p = 3, max_q = 3, m = 12,
                          start_P = 0, seasonal = True,
                          d = None, D = 1, trace = True,
                          error_action ='ignore',
                          suppress_warnings = True,
                          stepwise = True) 
    train = data.iloc[:len(data)-12]
    test = data.iloc[len(data)-12:] 
    model = SARIMAX(train['ticket_count'], 
                order = (1,0,0), 
                seasonal_order =(0, 1, 0, 12))
    result = model.fit()
    start = len(train)
    end = len(train) + len(test) - 1
    predictions = result.predict(start, end,typ = 'levels').rename("Predictions")
    rmse_error=rmse(test["ticket_count"], predictions)
    mse=mean_squared_error(test["ticket_count"], predictions)
    return rmse_error


# In[7]:


def lstm(data):
    results = seasonal_decompose(data['ticket_count'])
    train = data.iloc[:48]
    test = data.iloc[48:]
    test1=test.copy()
    scaler = MinMaxScaler()
    scaler.fit(train)
    scaled_train = scaler.transform(train)
    scaled_test = scaler.transform(test)
    scaled_train[:10]
    n_input = 3
    n_features = 1
    generator = TimeseriesGenerator(scaled_train, scaled_train, length=n_input, batch_size=1)
    X,y = generator[0]
    n_input = 12
    generator = TimeseriesGenerator(scaled_train, scaled_train, length=n_input, batch_size=1)
    model = Sequential()
    model.add(LSTM(100, activation='relu', input_shape=(n_input, n_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(generator,epochs=50)
    loss_per_epoch = model.history.history['loss']
    last_train_batch = scaled_train[-12:]
    last_train_batch = last_train_batch.reshape((1, n_input, n_features))
    model.predict(last_train_batch)
    test_predictions = []

    first_eval_batch = scaled_train[-n_input:]
    current_batch = first_eval_batch.reshape((1, n_input, n_features))
    for i in range(len(test)):
        current_pred = model.predict(current_batch)[0]
        test_predictions.append(current_pred)
        current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
    true_predictions = scaler.inverse_transform(test_predictions)
    test['Predictions'] = true_predictions
    rmse=sqrt(mean_squared_error(test1['ticket_count'],test['Predictions']))
    return rmse


# In[8]:


x=arima_sarima(data)


# In[9]:


y=lstm(data)


# In[10]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
data['Time'] = np.arange(len(data.index))
def linear_reg(data):
    def lin_reg(X,y):
        model = LinearRegression()
        model.fit(X, y)
        y_pred = pd.Series(model.predict(X), index=X.index)
        return y_pred
    X = data.loc[:, ['Time']]  
    y = data.loc[:, 'ticket_count'] 
    y_predi=lin_reg(X,y)
    data["predicted"]=y_predi
    return np.sqrt(mean_squared_error(y, y_predi))


# In[11]:


k=linear_reg(data)


# In[12]:


print('rmse of arima-sarima prediction is:',x)
print('rmse of lstm prediction is:',y)
print('rmse of Linear regression prediction is:',k)
if(x<y):
    print('the best model fit is arima-sarima:')
    model = model = SARIMAX(data['ticket_count'], 
                        order = (1, 0,0), 
                        seasonal_order =(0, 1, 0, 12))
    result = model.fit()
    forecast = result.predict(start = len(data), 
                          end = (len(data)-1) + 3 * 12, 
                          typ = 'levels').rename('Forecast')
    data['ticket_count'].plot(figsize = (12, 5), legend = True)
    plt.title('The forecasted graph for next 3years')
    forecast.plot(legend = True)


# In[13]:


forecast


# In[ ]:





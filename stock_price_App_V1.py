#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np
import pandas as pd
import streamlit as st
import pickle
from pickle import load
from textblob import TextBlob
from sklearn.preprocessing import StandardScaler


# Streamlit code 

# In[14]:


st.write("""
# This is Stock Price Prediction App
This app predicts the **Stock Price** for Future Expectancy!
""")


# In[15]:


st.sidebar.header('User Input Parameters')
def user_input_features():
    top=st.sidebar.text_input('Enter some text here', 'Enter newspapers topics here')
    openv=st.sidebar.number_input('Open', 0.0000,format="%.6f")
    high=st.sidebar.number_input('high', 0.0000,format="%.6f")
    low=st.sidebar.number_input('low', 0.0000,format="%.6f")
    close=st.sidebar.number_input('close', 0.0000,format="%.6f")
    volume=st.sidebar.number_input('volume', 0.0000,format="%.6f")
    adj_close=st.sidebar.number_input('Adj_close', 0.0000,format="%.6f")
    data = {'Top':top,
            'Open': openv,
            'High': high,
            'Low': low,
            'Close': close,
            'Volume':volume,
            'Adj Close':adj_close }
    feature = pd.DataFrame(data, index=[0])
    return feature

data = user_input_features()
st.subheader('User Input parameters')
st.write(data)
data["polarity"] = data["Top"].map(lambda a: TextBlob(a).sentiment[0])
data["subjectivity"] = data["Top"].map(lambda a: TextBlob(a).sentiment[1])
data.drop(["Top"], axis='columns', inplace=True)

data_ps= {'polarity':data["polarity"],
           'subjectivity':data["subjectivity"],
            'Open': data['Open'],
            'High': data['High'],
            'Low': data['Low'],
            'Close': data['Close'],
            'Volume':data['Volume'],
            'Adj Close':data['Adj Close']}
data_new = pd.DataFrame(data_ps, index=[0])
# load the scaler and scale the input data
#scaler = pickle.load(open('scaler.pkl', 'rb'))
#data_scaled=scaler.transform(data)
#test polarity and subjectivity
#st.subheader('User Input parameters')
#st.write(data_new)
#st.subheader('polarity')
#st.write(data["polarity"])
#st.subheader('subjectivity')
#st.write(data["subjectivity"])

#scaling input data 
x_train=np.load('xtrain.npy')
scaler2 = StandardScaler().fit(x_train)
rescaledX2 = scaler2.transform(x_train)
rescaledinputData = scaler2.transform(data_new)

#st.subheader('xtrain after scaling')
#st.write(rescaledX2[0])

#st.subheader('User Input parameters after scaling')
#st.write(rescaledinputData)

# Reads in saved classification model
load_clf = pickle.load(open('stock_price_clf.pkl', 'rb'))
predictions=load_clf.predict(rescaledinputData)

#st.subheader('The prediction of stock prices:(1 indicate increasing while 0 decreasing) ')
st.markdown("________")
st.info('The prediction of stock prices:1 indicate increasing while 0 decreasing. ')

st.subheader("This prediction is based on your input data: ")
st.write(predictions[0])



# In[ ]:





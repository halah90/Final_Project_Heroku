#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np
import pandas as pd
import streamlit as st
import pickle
from textblob import TextBlob


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
    openv=st.sidebar.number_input('Open', 0.0)
    high=st.sidebar.number_input('high', 0.0)
    low=st.sidebar.number_input('low', 0.0)
    close=st.sidebar.number_input('close', 0.0)
    volume=st.sidebar.number_input('volume', 0.0)
    adj_close=st.sidebar.number_input('Adj_close', 0.0)
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
data["subjectivity"] = data[f"Top"].map(lambda a: TextBlob(a).sentiment[1])
data.drop(["Top"], axis='columns', inplace=True)

# Reads in saved classification model
load_clf = pickle.load(open('stock_price_clf.pkl', 'rb'))
predictions=load_clf.predict(data)

st.subheader('1 means that Stock Price will increase, 0 will decrease ')
st.write(predictions)



# In[ ]:





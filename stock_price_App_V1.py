#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np
import pandas as pd
import streamlit as st
import pickle
from textblob import TextBlob
from sklearn.decomposition import PCA


# Streamlit code 

# In[14]:


st.write("""
# This is Stock Price Prediction App
This app predicts the **Stock Price** for Future Expectancy!
""")


# In[15]:


st.sidebar.header('User Input Parameters')
def user_input_features():
    top = st.sidebar.text_input('Enter some text here', 'Enter newspapers topics here')
    openv = st.sidebar.number_input('Open', 0.0,format="%.6f")
    close = st.sidebar.number_input('Close', 0.0,format="%.6f")
    data = {'Top':top,
            'Open': openv,
            'Close': close }
    feature = pd.DataFrame(data, index=[0])
    return feature

data = user_input_features()
st.subheader('User Input parameters')
st.write(data)
data["polarity"] = data["Top"].map(lambda a: TextBlob(a).sentiment[0])
data["subjectivity"] = data[f"Top"].map(lambda a: TextBlob(a).sentiment[1])
data.drop(["Top"], axis='columns', inplace=True)

data_to_pred1 = {'polarity': data["polarity"],
                'subjectivity' : data["subjectivity"],
                'Open': data['Open'],
                'Close': data['Close'] }
data_to_pred=pd.DataFrame(data_to_pred1, index=[0])

# Apply PCA
X_train2 = np.load('X_train2.npy')
pca = PCA(n_components=3)
pca.fit(X_train2)

transformed = pca.transform(data_to_pred)
pca_df = pd.DataFrame(transformed)

# Reads in saved classification model
load_clf = pickle.load(open('stock_price_xg.pkl', 'rb'))
predictions=load_clf.predict(pca_df)

#st.subheader('1 means that Stock Price will increase, 0 will decrease ')
st.markdown("________")
st.info('The prediction of stock prices:1 indicate increasing while 0 decreasing. ')

st.subheader("This prediction is based on your input data: ")
st.write(predictions[0])
import streamlit as st
import numpy as np
import pandas as pd
import joblib
model = joblib.load("iris_model.joblib")
st.title('Flower prediction')
columns = ['sepal length (cm)',
 'sepal width (cm)',
 'petal length (cm)',
 'petal width (cm)']
label_dict = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
sepal_length = st.number_input('Input length of sepal')
sepal_width = st.number_input('Input width of sepal ')
petal_lenth = st.number_input('Input lenth of petal')
petal_width = st.number_input('Input width of petal')

def predict():
    row = np.array([sepal_length,sepal_width,petal_lenth,petal_width])
    X= pd.DataFrame([row],columns=columns)
    prediction = model.predict(X)[0]
    st.success(f'Predicted flower is: {label_dict[prediction]}')
st.button('Predict',on_click=predict)
import pandas as pd
import xgboost as xgb
import os
from pyprojroot import here
import numpy as np
import joblib
import streamlit as st

path_outputs = here("./outputs")
os.chdir(path_outputs)

model_iu_bball = joblib.load("model_iu_bball.jlib")


st.title('Prediction app for IU making the NCAA tournament')

w = st.number_input('Enter wins:', min_value=0.0, step=0.01, value = 22.0)
l = st.number_input('Enter losses:', min_value=0.0, step=0.01, value = 12.0)
w_c = st.number_input('Enter conference wins:', min_value=0.0, step=0.01, value = 11.0)
l_c = st.number_input('Enter conference losses:', min_value=0.0, step=0.01, value = 8.0)
sos = st.number_input('Enter strength of schedule (sos):', min_value=0.0, step=0.01, value = 10.0)
ps_g = st.number_input('Enter points per game:', min_value=0.0, step=0.01, value = 75.0)
pa_g = st.number_input('Enter points allowed per game:', min_value=0.0, step=0.01, value = 67.0)
ap_pre = st.number_input('Enter AP preseason ranking:', min_value=0.0, step=0.01, value = 15.0)
ap_final = st.number_input('Enter AP high ranking:', min_value=0.0, step=0.01, value = 22.0)
input_data = np.array([w, l, w_c, l_c, sos, ps_g, pa_g, ap_pre, ap_final]).reshape(1, -1)

if st.button('Predict'):
    prediction = model_iu_bball.predict(input_data)
    percentage_prediction = round(prediction[0] * 100, 0)
    st.write(f'The prediction is: {percentage_prediction}%')



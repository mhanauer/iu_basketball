import pandas as pd
import os
from pyprojroot import here
import numpy as np
from skimpy import clean_columns
import joblib

path_outputs = here("./outputs")
os.chdir(path_outputs)

model_iu_bball = joblib.load("model_iu_bball.jlib")

import streamlit as st
import numpy as np
import pandas as pd
import joblib

st.title('Prediction app for IU making the NCAA tournament')

w = st.number_input('Enter wins:', min_value=0.0, step=0.01, value = 15.0)
l = st.number_input('Enter losses:', min_value=0.0, step=0.01, value = 15.0)
w_c = st.number_input('Enter conference wins:', min_value=0.0, step=0.01, value = 10.0)
l_c = st.number_input('Enter conference losses:', min_value=0.0, step=0.01, value = 10.0)
sos = st.number_input('Enter strength of schedule (sos):', min_value=0.0, step=0.01, value = 20.0)
ps_g = st.number_input('Enter points per game:', min_value=0.0, step=0.01, value = 75.0)
pa_g = st.number_input('Enter points allowed per game:', min_value=0.0, step=0.01, value = 70.0)
ap_pre = st.number_input('Enter AP preseason ranking:', min_value=0.0, step=0.01, value = 25.0)
ap_final = st.number_input('Enter AP high ranking:', min_value=0.0, step=0.01, value = 25.0)
input_data = np.array([w, l, w_c, l_c, sos, ps_g, pa_g, ap_pre, ap_final]).reshape(1, -1)

if st.button('Predict'):
    prediction = model_iu_bball.predict(input_data)
    rounded_prediction = round(prediction[0], 2)
    st.write(f'The prediction is: {rounded_prediction}')


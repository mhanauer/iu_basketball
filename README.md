IU NCAA Tournament Prediction App
This is a simple Streamlit app that predicts the probability of Indiana University (IU) men's basketball team making the NCAA tournament based on various input features. The model used in this app is a LightGBM model trained on historical data.

Requirements
This app requires the following libraries to be installed:

pandas
os
lightgbm
pyprojroot
numpy
joblib
streamlit
These can be installed by running the command pip install -r requirements.txt in the command line.

Usage
To use this app, simply run the following command in the command line:

streamlit run app.py

Then, enter the required input features such as wins, losses, conference wins, conference losses, strength of schedule, points per game, points allowed per game, AP preseason ranking, and AP high ranking. The default values for each feature have been provided, but they can be changed to fit the desired scenario. Once the input values have been entered, click on the "Predict" button to generate the prediction.

Files
app.py: This is the main file containing the Streamlit app code.
model_iu_bball.jlib: This is the trained LightGBM model used in the app.
requirements.txt: This file contains the required libraries to run the app.

Disclaimer
This app is intended for educational purposes only and should not be used as a basis for any real-world decisions. The prediction is based on a trained model and may not accurately reflect the actual probability of IU making the NCAA tournament.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
df = pd.read_csv('liver.csv')
#df = pd.read_csv('cleaned_liver_disease_dataset.csv')
X = df.drop('Result', axis=1)
y = df['Result']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier()
model.fit(X_train, y_train)


import streamlit as st
import numpy as np
import pickle

# Load the trained model
#
#model = load_model()

# Define the function to make predictions
def predict_liver_disease(age, gender, total_bilirubin, direct_bilirubin,
                          alkaline_phosphotase, alamine_aminotransferase,
                          aspartate_aminotransferase, total_proteins,
                          albumin, albumin_and_globulin_ratio):
    features = np.array([
        [age, gender, total_bilirubin, direct_bilirubin,
         alkaline_phosphotase, alamine_aminotransferase,
         aspartate_aminotransferase, total_proteins,
         albumin, albumin_and_globulin_ratio]
    ])
    prediction = model.predict(features)
    return prediction[0]

# Streamlit app
st.title('Liver Disease Prediction')

# Collect user input
age = st.number_input('Age', min_value=0)
gender = st.selectbox('Gender', options=['Female', 'Male'])
total_bilirubin = st.number_input('Total Bilirubin', min_value=0.0)
direct_bilirubin = st.number_input('Direct Bilirubin', min_value=0.0)
alkaline_phosphotase = st.number_input('Alkaline Phosphotase', min_value=0)
alamine_aminotransferase = st.number_input('Alamine Aminotransferase', min_value=0)
aspartate_aminotransferase = st.number_input('Aspartate Aminotransferase', min_value=0)
total_proteins = st.number_input('Total Proteins', min_value=0.0)
albumin = st.number_input('Albumin', min_value=0.0)
albumin_and_globulin_ratio = st.number_input('Albumin and Globulin Ratio', min_value=0.0)

# Button to make predictions
if st.button('Predict'):
    gender_code = 1 if gender == 'Male' else 0  # Convert gender to 1 or 0
    prediction = predict_liver_disease(age, gender_code,
                                       total_bilirubin, direct_bilirubin,
                                       alkaline_phosphotase, alamine_aminotransferase,
                                       aspartate_aminotransferase, total_proteins,
                                       albumin, albumin_and_globulin_ratio)
    if prediction == 1:
        st.write('The model predicts that the patient has liver disease.')
    else:
        st.write('The model predicts that the patient does not have liver disease.')



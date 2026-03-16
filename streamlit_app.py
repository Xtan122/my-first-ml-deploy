import streamlit as st
import joblib

# Load model và scaler
model = joblib.load('linear_regression_model.pkl')
scaler = joblib.load('scaler.pkl')

# Streamlit app
st.title('Student Test Score Prediction')
st.write('Enter the number of hours studied to predict the test score.')

# User input
hours = st.number_input('Hours Studied', min_value=0.0, step=1.0)

# Predict
if st.button('Predict'):
    try:
        data = [[hours]]
        scaled_data = scaler.transform(data)
        prediction = model.predict(scaled_data)[0]

        st.success(f"Predicted Test Score: {prediction[0]:.2f}")

    except Exception as e:
        st.error(f'Error: {e}')
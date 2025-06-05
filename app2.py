import pickle
import pandas as pd
import streamlit as st

# Load the pre-trained model
with open('rand_model.pkl', 'rb') as file:
    model = pickle.load(file)

def preprocess_input(hr, weekday, temp, atemp, hum, windspeed, season, yr, mnth, holiday, workingday, weathersit):
    data = {
        'hr': [hr],
        'weekday': [weekday],
        'temp': [temp],
        'atemp': [atemp],
        'hum': [hum],
        'windspeed': [windspeed],
        'season_fall': [1 if season == 'fall' else 0],
        'season_spring.0': [1 if season == 'spring' else 0],  # Updated name
        'season_summer': [1 if season == 'summer' else 0],
        'season_winter': [1 if season == 'winter' else 0],
        'yr_2011.0': [1 if yr == 2011 else 0],  # Updated name
        'yr_2012.0': [1 if yr == 2012 else 0],  # Updated name
        'mnth_' + str(mnth) + '.0': [1],  # Adjusted to match `mnth_1.0`, etc.
        'holiday_No': [1 if holiday == 0 else 0],
        'holiday_Yes': [1 if holiday == 1 else 0],
        'workingday_No work': [1 if workingday == 0 else 0],
        'workingday_Working Day': [1 if workingday == 1 else 0],
        'weathersit_Clear': [1 if weathersit == 1 else 0],
        'weathersit_Heavy Rain': [1 if weathersit == 3 else 0],
        'weathersit_Light Snow': [1 if weathersit == 4 else 0],
        'weathersit_Mist': [1 if weathersit == 2 else 0],
    }
    return pd.DataFrame(data)

def align_features(input_df, model):
    # Get expected feature names from the model
    expected_features = model.feature_names_in_

    # Add missing columns with default value 0
    for col in expected_features:
        if col not in input_df:
            input_df[col] = 0

    # Ensure column order matches the model
    input_df = input_df[expected_features]

    return input_df

def predict(input_data):
    prediction = model.predict(input_data)
    return prediction[0]  # Assuming the model returns a single value

def main():
    st.title('Bike-Sharing Rental Prediction')

    with st.form("prediction_form"):
        hr = st.number_input('Hour (0 to 23):', min_value=0, max_value=23, step=1)
        weekday = st.number_input('Weekday (0 to 6):', min_value=0, max_value=6, step=1)
        temp = st.number_input('Temperature (0 to 1):', min_value=0.0, max_value=1.0, step=0.01)
        atemp = st.number_input('Feels Like Temperature (0 to 1):', min_value=0.0, max_value=1.0, step=0.01)
        hum = st.number_input('Humidity (0 to 1):', min_value=0.0, max_value=1.0, step=0.01)
        windspeed = st.number_input('Wind Speed (0 to 1):', min_value=0.0, max_value=1.0, step=0.01)
        season = st.selectbox('Season:', options=['spring', 'summer', 'fall', 'winter'])
        yr = st.selectbox('Year:', options=[2011, 2012])
        mnth = st.number_input('Month (1 to 12):', min_value=1, max_value=12, step=1)
        holiday = st.selectbox('Holiday:', options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
        workingday = st.selectbox('Working Day:', options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
        weathersit = st.number_input('Weather Situation (1 to 4):', min_value=1, max_value=4, step=1)

        submitted = st.form_submit_button("Predict")

        if submitted:
            # Preprocess inputs
            input_data = preprocess_input(hr, weekday, temp, atemp, hum, windspeed, season, yr, mnth, holiday, workingday, weathersit)
            input_data = align_features(input_data, model)
            
            # Perform prediction
            prediction = predict(input_data)
            st.success(f"Predicted Bike Rentals: {int(prediction)}")

if __name__ == '__main__':
    main()

import pickle
import streamlit as st
import pandas as pd
import gdown
import os

# Define the URL for the Google Drive file
model_url = 'https://drive.google.com/uc?id=113nMMGbKmHdd__4UpYeXoK_i4GDq2_Kg'
model_path = 'random_forest_model.pkl'

# Download the model file if it doesn't already exist
if not os.path.exists(model_path):
    gdown.download(model_url, model_path, quiet=False)

# Load the trained Random Forest model
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Function to make predictions
def predict(input_data):
    prediction = model.predict(input_data)
    return prediction

# Streamlit app
st.title('Road Collision Severity Predictor')

st.sidebar.header('User Input Features')
st.sidebar.markdown('Please input the features below:')

# Collect user input features
def user_input_features():
    number_of_vehicles = st.sidebar.selectbox('Number of Vehicles', [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 17])
    number_of_casualties = st.sidebar.selectbox('Number of Casualties', [1, 2, 3, 4, 5, 6, 7, 8, 12, 17, 19])
    day_of_week = st.sidebar.selectbox('Day of Week', [1, 2, 3, 4, 5, 6, 7])
    first_road_class = st.sidebar.selectbox('First Road Class', [1, 2, 3, 4, 5, 6])
    first_road_number = st.sidebar.number_input('First Road Number', value=0)
    road_type = st.sidebar.selectbox('Road Type', [1, 2, 3, 6, 7, 9])
    speed_limit = st.sidebar.selectbox('Speed Limit', [20, 30, 40, 50, 60, 70])
    junction_detail = st.sidebar.selectbox('Junction Detail', [0, 1, 2, 3, 5, 6, 7, 8, 9, 99])
    junction_control = st.sidebar.selectbox('Junction Control', [1, 2, 3, 4, 9])
    light_conditions = st.sidebar.selectbox('Light Conditions', [1, 4, 5, 6, 7])
    weather_conditions = st.sidebar.selectbox('Weather Conditions', [1, 2, 3, 4, 5, 6, 7, 8, 9])
    road_surface_conditions = st.sidebar.selectbox('Road Surface Conditions', [1, 2, 3, 4, 5, 9])
    special_conditions_at_site = st.sidebar.selectbox('Special Conditions at Site', [0, 1, 2, 3, 4, 5, 6, 7, 9])
    pedestrian_crossing_human_control = st.sidebar.selectbox('Pedestrian Crossing Human Control', [0, 1, 2, 9])
    pedestrian_crossing_physical_facilities = st.sidebar.selectbox('Pedestrian Crossing Physical Facilities', [0, 1, 4, 5, 7, 8, 9])
    did_police_officer_attend_scene_of_collision = st.sidebar.selectbox('Did Police Officer Attend Scene', [0, 1])

    data = {
        'number_of_vehicles': number_of_vehicles,
        'number_of_casualties': number_of_casualties,
        'day_of_week': day_of_week,
        'first_road_class': first_road_class,
        'first_road_number': first_road_number,
        'road_type': road_type,
        'speed_limit': speed_limit,
        'junction_detail': junction_detail,
        'junction_control': junction_control,
        'light_conditions': light_conditions,
        'weather_conditions': weather_conditions,
        'road_surface_conditions': road_surface_conditions,
        'special_conditions_at_site': special_conditions_at_site,
        'pedestrian_crossing_human_control': pedestrian_crossing_human_control,
        'pedestrian_crossing_physical_facilities': pedestrian_crossing_physical_facilities,
        'did_police_officer_attend_scene_of_collision': did_police_officer_attend_scene_of_collision
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Display user input
st.subheader('User Input Features')
st.write(input_df)

# Predict button
if st.button('Predict'):
    prediction = predict(input_df)
    st.subheader('Prediction')
    st.write(f'Collision Severity: {prediction[0]}')

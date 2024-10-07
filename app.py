import os
import pickle
import streamlit as st
import pandas as pd

# Function to save the model and feature names
def save_model(model, feature_names, model_path='models/decision_tree_model.pkl'):
    # Ensure the 'models' directory exists in the current working directory
    if not os.path.exists('models'):
        os.makedirs('models')
        
    # Save the model and feature names
    with open(model_path, 'wb') as file:
        pickle.dump((model, feature_names), file)
    print("Model and feature names saved successfully!")

# Load the model and feature names
def load_model(model_path='models/decision_tree_model.pkl'):
    if os.path.exists(model_path):
        with open(model_path, 'rb') as file:
            loaded_model, feature_names = pickle.load(file)
        return loaded_model, feature_names
    else:
        raise FileNotFoundError("Model file not found. Please check the model path.")

# Load the model and feature names
try:
    loaded_model, feature_names = load_model()
    print("Model and feature names loaded successfully!")
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()  # Stop the execution if the model is not found

# Set up Streamlit app
st.set_page_config(page_title="Cancer Prediction", layout="wide")

# Streamlit app title
st.title('Cancer Prediction App')

st.write("""## Enter the patient information:""")

# Input fields for user habits
def user_input_features():
    smoking_habit = st.selectbox('Smoking Habit', options=['Heavy', 'Moderate', 'Occasional', 'Non-Smoker'])
    drinking_habit = st.selectbox('Drinking Habit', options=['Frequent', 'Moderate', 'Occasional', 'Non-Drinker'])
    biking_habit = st.selectbox('Biking Habit', options=['Low', 'Medium', 'High'])
    walking_habit = st.selectbox('Walking Habit', options=['Low', 'Medium', 'High'])
    jogging_habit = st.selectbox('Jogging Habit', options=['Low', 'Medium', 'High'])

    data = {
        'Smoking Habit': smoking_habit,
        'Drinking Habit': drinking_habit,
        'Biking Habit': biking_habit,
        'Walking Habit': walking_habit,
        'Jogging Habit': jogging_habit
    }

    features = pd.DataFrame(data, index=[0])
    return features

# Get user input
input_df = user_input_features()

# Display the user input
st.write("### Patient's information:")
st.write(input_df)

# Prediction button
if st.button('Predict'):
    # Make predictions
    if 'loaded_model' in locals():  # Ensure the model is loaded
        # One-hot encode the features
        input_df_encoded = pd.get_dummies(input_df, drop_first=True).reindex(columns=feature_names, fill_value=0)
        
        prediction = loaded_model.predict(input_df_encoded)
        prediction_proba = loaded_model.predict_proba(input_df_encoded)

        # Display the prediction
        st.write("### Prediction")
        if prediction[0] == 1:
            st.write("Patient has a high risk of cancer.")
        else:
            st.write("Patient has a low risk of cancer.")

        # Display the prediction probabilities
        st.write("### Prediction Probability")
        st.write(f"Probability of having cancer: {prediction_proba[0][1]:.2f}")
        st.write(f"Probability of not having cancer: {prediction_proba[0][0]:.2f}")
    else:
        st.error("Model not loaded, cannot make predictions.")

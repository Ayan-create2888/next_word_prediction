import streamlit as st
import requests

st.title("Next Word Prediction System")

# Input text
user_input = st.text_input("Enter a sequence of words:", "to be or")

if st.button("Predict Next Word"):
    if user_input.strip():
        # Call FastAPI endpoint
        response = requests.post("https://http://127.0.0.1:8000/predict", json={"text": user_input})
        if response.status_code == 200:
            result = response.json()
            predicted_word = result.get("predicted_word", "No prediction")
            st.success(f"Predicted next word: **{predicted_word}**")
        else:
            st.error("Error: Could not get prediction from server.")
    else:

        st.warning("Please enter some text.")

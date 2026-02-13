import streamlit as st
import requests

st.title("Next Word Prediction System")

# Input text
user_input = st.text_input("Enter a sequence of words:", "to be or")

if st.button("Predict Next Word"):
    if user_input.strip():
        # Call FastAPI endpoint
        response = requests.post("https://ayyan2009-api-netword.hf.space/predict", json={"text": user_input})
        if response.status_code == 200:
            result = response.json()
            predicted_word = result.get("predicted_word", "No prediction")
            st.success(f"Predicted next word: **{predicted_word}**")
        else:
            st.error("Error: Could not get prediction from server.")
    else:
        st.warning("Please enter some text.")

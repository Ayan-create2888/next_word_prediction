from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
import os
from datetime import datetime

app = FastAPI()

# Load model and tokenizer
model = tf.keras.models.load_model('next_word_model.h5')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

max_sequence_len = 20  # Match from training
DATA_FILE = "data.json"

class TextInput(BaseModel):
    text: str


def save_to_json(input_text: str, output_text: str):
    new_entry = {
        "input": input_text,
        "output": output_text,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # If file exists, load existing data
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            data = json.load(f)
    else:
        data = []

    # Append new entry
    data.append(new_entry)

    # Save back to file
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=4)


@app.post("/predict")
def predict_next_word(input_data: TextInput):
    seed_text = input_data.text.lower()

    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences(
        [token_list],
        maxlen=max_sequence_len - 1,
        padding='pre'
    )

    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)[0]
    predicted_word = tokenizer.index_word.get(predicted_word_index, "")

    # Save input & output to JSON file
    save_to_json(seed_text, predicted_word)

    return {"predicted_word": predicted_word}


@app.get("/")
def read_root():
    return {"message": "Next Word Prediction API"}

from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = FastAPI()

# Load model and tokenizer
model = tf.keras.models.load_model('next_word_model.h5')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

max_sequence_len = 20  # Match from training

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict_next_word(input_data: TextInput):
    seed_text = input_data.text.lower()
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)[0]
    predicted_word = tokenizer.index_word.get(predicted_word_index, "")
    return {"predicted_word": predicted_word}

@app.get("/")
def read_root():
    return {"message": "Next Word Prediction API"}
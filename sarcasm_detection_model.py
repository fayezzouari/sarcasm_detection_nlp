import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load the model
model = tf.keras.models.load_model("sarcasm_model.h5")

# Load the tokenizer
with open("tokenizer.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)

# Predict on new sentences yet there is a problem with one sentence, it is supposed to be sarcastic but it is not according to the model
# This is because the model was trained on a dataset that had a lot of sentences that were not sarcastic
sentences = [
    "granny starting to fear spiders in the garden might be real",
    "game of thrones season finale showing this sunday night",
    "I am busy right now, can I ignore you some other time?",
]

sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, padding="post", maxlen=100)
predictions = model.predict(padded)
print(predictions)

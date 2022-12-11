import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import numpy as np
import json
import random
import string
import pandas as pd
from flask import Flask, render_template, request
# import tensorflow as tf
import os


### Import model file ###
MODEL_DIR = os.path.join(os.path.dirname('__file__'), 'biD_lstm_model3.h5')
model = load_model(MODEL_DIR)


with open("Intent.json") as train_file:
    df = json.load(train_file)

tags = []
inputs = []
responses = {}

for intent in df['intents']:
    responses[intent['intent']] = intent['responses']
    for lines in intent['text']:
        inputs.append(lines)
        tags.append(intent['intent'])

df = pd.DataFrame({'inputs': inputs, 'tags': tags})
df

# removing punctuation
df["inputs"] = df["inputs"].apply(
    lambda wrd: [ltrs.lower() for ltrs in wrd if ltrs not in string.punctuation])
df["inputs"] = df["inputs"].apply(lambda wrd: "".join(wrd))
df

# Tokenizing data
tokenizer = Tokenizer(num_words=2000)
tokenizer.fit_on_texts(df["inputs"])
train = tokenizer.texts_to_sequences(df["inputs"])
tokens = tokenizer.texts_to_sequences(df["inputs"])

# apply padding
x_train = pad_sequences(train)

# encoding the outputs
le = LabelEncoder()
y_train = le.fit_transform(df['tags'])

input_shape = x_train.shape[1]


def test(msg):

    texts_p = []
    prediction_input = msg

    # removing punctuation and converting to lowercase
    prediction_input = [letters.lower(
    ) for letters in prediction_input if letters not in string.punctuation]
    prediction_input = "".join(prediction_input)
    texts_p.append(prediction_input)

    #tokenizing and padding
    prediction_input = tokenizer.texts_to_sequences(texts_p)
    prediction_input = np.array(prediction_input).reshape(-1)
    prediction_input = pad_sequences([prediction_input], input_shape)

    # getting output from model
    output = model.predict(prediction_input)
    output = output.argmax()

    # finding the right tag and predicting
    response_tag = le.inverse_transform([output])[0]
    bot_answer = random.choice(responses[response_tag])
    return bot_answer


# from flask import Flask, render_template, request
app = Flask(__name__)
app.static_folder = 'static'


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return test(userText)


if __name__ == "__main__":
    app.run

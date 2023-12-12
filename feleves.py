import streamlit as st
# For ML Models
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.losses import *
from tensorflow.keras.models import *
from tensorflow.keras.metrics import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.applications import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# For Data Processing
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import *

# For Data Visualization
import matplotlib.pyplot as plt
import matplotlib
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns


# Miscellaneous





df = pd.read_csv('dataset.csv')

tokenizer = Tokenizer(filters='"&(),-/:;<=>[\\]_`{|}~\t\n0123456789',
                      lower=True, split=' ')
tokenizer.fit_on_texts(np.array(df['text']))
vocab_size = len(tokenizer.word_index) + 1

lengths=[]
for x in range(len(df)):
    i = df['text'][x]
    i = tokenizer.texts_to_sequences([i])
    lengths.append(len(i[0]))
lengths = np.array(lengths)

maxlen = 15

df['humor'] = df['humor'].apply(lambda x: {True:1, False:0}.get(x))
texts = np.array(df['text'])
texts = tokenizer.texts_to_sequences(texts)
for x in range(len(texts)):
    if len(texts[x])>maxlen:
        texts[x]=texts[x][:maxlen]
texts = pad_sequences(texts, maxlen=maxlen, dtype='float', padding='post', value=0.0)
texts = np.array(texts)
labels = df['humor']
labels = np.array([float(j) for j in labels])

x_train, x_val, y_train, y_val = train_test_split(texts, labels, test_size=0.2, random_state=0)

model = Sequential()
model.add(Input(shape=(maxlen)))
model.add(Embedding(vocab_size, 8, input_length=maxlen))
#model.add(Bidirectional(LSTM(16, activation='relu',dropout=0.15, return_sequences=True), merge_mode='concat'))
#model.add(TimeDistributed(Dense(16, activation='relu')))
model.add(LSTM(64, activation='relu',dropout=0.15, return_sequences=False))
model.add(Flatten())
#model.add(Dense(64, activation='relu'))
#model.add(Dropout(0.25))
model.add(Dense(2, activation='softmax'))

model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy',
             metrics=['sparse_categorical_accuracy'])

epochs = 10
#mc = ModelCheckpoint('model.h5', monitor='val_sparse_categorical_accuracy', mode='max', verbose=1, save_best_only=True)
history = model.fit(x_train, y_train, epochs=epochs, batch_size=8192, validation_data=(x_val, y_val))

def predict(text):
    text = tokenizer.texts_to_sequences([text])
    if len(text)>maxlen:
        text=text[:maxlen]
    text = pad_sequences(text, maxlen=maxlen, dtype='float', padding='post', value=0.0)
    text = np.array(text)
    pred = model.predict(text)
    pred = np.argmax(pred, axis=-1)
    decode_label = {0:'False', 1:'True'}
    pred = decode_label[pred[0]]
    return pred

st.write("Enter a funny joke")

imp=st.text_input(label='Joke',value="", max_chars=None, key=None, type="default", help=None, autocomplete="Write your joke here", on_change=None, args=None, kwargs=None, placeholder=None, disabled=False, label_visibility="visible")


pred = predict(imp)
print("Text:",imp)
print('Humor detected: ',pred)
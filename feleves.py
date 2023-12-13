import streamlit as st

# For ML Models
import json
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
from tensorflow.keras.preprocessing.text import tokenizer_from_json

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


@st.cache_resource
def create_tokenizer():

    df = pd.read_csv('dataset.csv')
    tokenizer = Tokenizer(filters='"&(),-/:;<=>[\\]_`{|}~\t\n0123456789',
                      lower=True, split=' ')
    tokenizer.fit_on_texts(np.array(df['text']))
    vocab_size = len(tokenizer.word_index) + 1

create_tokenizer()
maxlen = 15







#x_train, x_val, y_train, y_val = train_test_split(texts, labels, test_size=0.2, random_state=0)


model = load_model('modello.h5')

#epochs = 10
#mc = ModelCheckpoint('model.h5', monitor='val_sparse_categorical_accuracy', mode='max', verbose=1, save_best_only=True)
#history = model.fit(x_train, y_train, epochs=epochs, batch_size=8192, validation_data=(x_val, y_val))



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


def pred(imp):

    
   pred = predict(imp)
   print(pred)
   if pred=='True':
        st.write('You are funny!')
    
   else:
        st.write('This was not funny!')

st.write("Enter a funny joke")

imp=st.text_input(label='Joke',value='', max_chars=300, type="default", help=None,  args=None, kwargs=None, placeholder="Write your joke here", disabled=False, label_visibility="hidden")
#lst=st.


#if imp:
 #   pred(imp)

but=st.button(label="Test my Joke", args=None, kwargs=None,  type="secondary", disabled=False, use_container_width=False)
box=st.selectbox("I don't want to enter my own joke: ",(" ","what's the difference between donald trump's hair and a wet racoon", "All pants are breakaway pants if you're angry enough","5 reasons the 2016 election feels so personal","Pasco police shot mexican migrant from behind, new autopsy shows"))
if but:

    if imp!=' ':
        pred(imp)



text = "what's the difference between donald trump's hair and a wet racoon"
pred = predict(text)
print("Text:",text)
ctr=1
print('Humor detected: ',pred)

   
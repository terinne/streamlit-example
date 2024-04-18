import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

# read the data and clean it
df_train = pd.read_csv("train.csv", encoding='unicode_escape')
df_test = pd.read_csv("test.csv", encoding='latin1')

df_train.drop(columns={'textID', 'selected_text', 'Time of Tweet', 'Age of User', 'Country', 'Population -2020', 'Land Area (Km²)', 'Density (P/Km²)'}, inplace=True)
df_train.dropna(inplace=True)

df_test.drop(columns={'textID', 'Time of Tweet', 'Age of User', 'Country', 'Population -2020', 'Land Area (Km²)', 'Density (P/Km²)'}, inplace=True)
df_test.dropna(inplace=True)

x_train = df_train['text']
x_test = df_test['text']
y_train = df_train['sentiment']
y_test = df_test['sentiment']

model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LinearSVC())
])

model.fit(x_train, y_train)

st.title('Sentiment Analysis App')
st.divider()
st.subheader('Insert your text, press submit and see the results!')

user_input = st.text_input('Text', label_visibility='collapsed', placeholder='Write something...')

def analyze():
    global user_input
    container_1.write(f'Text to analyze: {user_input}')
    user_input = ""

    result = model.predict([user_input])[0]

    if result == "positive":
        container_2.write(f'The result is {result}! :smile:')
    elif result == "neutral":
        container_2.write(f'The result is {result}! :neutral_face:')
    elif result == "negative":
        container_2.write(f'The result is {result}! :rage:')

st.button('SUBMIT', use_container_width=True, on_click=analyze)

container_1 = st.container(border=True)
container_2 = st.container(border=True)

    


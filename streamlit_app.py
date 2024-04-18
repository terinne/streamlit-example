import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

def analyze():
    container_1.write(f'Text to analyze: {st.session_state.user_input}')
    model = st.session_state.model
    st.session_state.result = model.predict([st.session_state.user_input])[0]

    current_result = st.session_state.result

    if current_result == "positive":
        container_2.write(f'The result is {st.session_state.result}! :smile:')
    elif current_result == "neutral":
        container_2.write(f'The result is {st.session_state.result}! :neutral_face:')
    elif current_result == "negative":
        container_2.write(f'The result is {st.session_state.result}! :rage:')
    
    st.session_state.user_input = ""

st.title('Sentiment Analysis App')
st.divider()
container_3 = st.container(border=True)

st.subheader('Insert your text, press submit and see the results!')

st.text_input('Text', key='user_input',  label_visibility='collapsed', placeholder='Write something...')

st.button('SUBMIT', use_container_width=True, on_click=analyze)

container_1 = st.container(border=True)
container_2 = st.container(border=True)


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

st.session_state.model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LinearSVC())
])

st.session_state.model.fit(x_train, y_train)

predictions = st.session_state.model.predict(x_test)
report = classification_report(y_test, predictions, output_dict=True)
container_3.write('The results of the training:')
container_3.dataframe(report)
    



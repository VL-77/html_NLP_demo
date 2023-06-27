import streamlit as st
import pandas as pd
from pages.fetch import *

from textblob import TextBlob
import sys
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image
import matplotlib.pyplot as plt
import spacy
import ru_core_news_sm
import en_core_web_sm
from spacy import load
import torch
from transformers import AutoTokenizer, AutoModelWithLMHead, GPT2LMHeadModel
import docx2txt
from PIL import Image
from PyPDF2 import PdfFileReader
from pdf2image import convert_from_bytes
import pdfplumber
#from line_cor import mark_region
import pdf2image
import numpy as np
np.bool = np.bool
# Function to Analyse Tokens and Lemma

@st.cache
def text_analyzer(my_text):
    nlp = spacy.load("ru_core_news_sm")
    docx = nlp(my_text)
    for docx in nlp.pipe(my_text, disable=["tagger", "parser", "entity_ruler", "sentencizer", "textcat"]):
        print([(ent.text, ent.label_) for ent in docx.ents])
    allData = [('"Токен":{},\n"Лемма":{},"Часть речи":{},\n"Ключевое слово":{},"Роль в отношении зависимости":{},\n"Обобщенная форма токена":{},"Состоит ли токен из букв":{},\n"Является ли стоп-словом":{}'.format(token.text, token.lemma_, token.pos_, token.tag_, token.dep_, token.shape_, token.is_alpha, token.is_stop)) for token in docx]
    return allData

def load_models():
    tokenizer = AutoTokenizer.from_pretrained('gpt2-large')
    model = GPT2LMHeadModel.from_pretrained('gpt2-large')
    return tokenizer, model
# Function For Extracting Entities


# FUnction for pos tagging
@st.cache
def pos_tagging(my_text):
    data = {}
    nlp = spacy.load("ru_core_news_sm")

    doc = nlp(my_text)

    c_tokens = [token.text for token in doc]
    c_pos = [token.pos_ for token in doc]
    c_lemma = [token.lemma_ for token in doc]
    c_stop = [token.is_stop for token in doc]
    new_df = pd.DataFrame(zip(c_tokens, c_lemma, c_pos, c_stop),
                          columns=['Tokens','Lemma','POS','STOP'])

    return new_df


# Function for sentiment analysis

def sent_analysis(my_text):
    testimonial = TextBlob(my_text)
    return testimonial.sentiment.polarity, testimonial.sentiment.subjectivity


# Function for world cloud

def word_cloud(my_text):
    wordcloud = WordCloud(width=1200, height=600, background_color='white', random_state=42,
                          stopwords=set(STOPWORDS)).generate(my_text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot()


def main():
    # Title
    front_up()
    st.title('Базовый NLP ')

    if st.checkbox('Показать токены и лемму', key='token'):
        st.subheader('Обозначьте свой текст')
        boool, text = selection(key='token')

        if st.button("Анализировать", key='token'):
            if boool == 0:
                message = text
                nlp_result = text_analyzer(message)
                st.json(nlp_result)
            else:
                try:
                    message = get_text(text)
                    nlp_result = text_analyzer(message)
                    st.json(nlp_result)
                except BaseException as e:
                    st.warning(e)

    if st.checkbox('Показывать части речи', key='pos'):
        st.subheader('Пометка POS в вашем тексте')

        boool, text = selection(key='pos')

        if st.button("Анализировать", key='pos'):
            if boool == 0:
                message = text
                nlp_result = pos_tagging(message)
                st.dataframe(nlp_result)
            # function here
            else:
                try:
                    message = get_text(text)
                    nlp_result = pos_tagging(message)
                    st.dataframe(nlp_result)
                # function here
                except BaseException as e:
                    st.warning(e)

    if st.checkbox('Покажите настроение предложения', key='sent'):
        st.subheader('Субъективность и полярность в тексте')

        boool, text = selection(key='sent')

        if st.button("Анализировать", key='sent'):
            if boool == 0:
                message = text
                polarity, subjectivity = sent_analysis(message)
                st.info("Полярность: {} , Субъективность: {} ".format(polarity, subjectivity))

            # function here
            else:
                try:
                    message = get_text(text)
                    polarity, subjectivity = sent_analysis(message)
                    st.info("Полярность: {} , Субъективность: {} ".f(polarity, subjectivity))
                # function here
                except BaseException as e:
                    st.warning(e)

    if st.checkbox('Показать облако слов', key='cloud'):
        st.subheader('Нанесите облако слов на текст')

        boool, text = selection(key='cloud')

        if st.button("Анализировать", key='cloud'):
            if boool == 0:
                message = text

                # function here
                word_cloud(message)
            else:
                try:
                    message = get_text(text)

                    # function here
                    word_cloud(message)
                except BaseException as e:
                    st.warning(e)




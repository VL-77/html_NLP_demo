import nltk
from gensim.corpora import dictionary
from gensim import corpora, models, similarities
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import punkt
from gensim import corpora
from gensim import models 
import gensim
import pyLDAvis.gensim
import pyLDAvis
from gensim.models import LdaModel
import re
import streamlit as st
import pandas as pd
from pages.fetch import *
from textblob import TextBlob
import sys
from matplotlib import pyplot as plt
import numpy as np
import spacy
from spacy import displacy
import codecs as cd
import gensim
import en_core_web_sm
import ru_core_news_sm
import xx_ent_wiki_sm
from spacy import load





def entity_analyzer(my_text):
    nlp = spacy.load("xx_ent_wiki_sm")
    docx = nlp(my_text)
    tokens = [token.text for token in docx]
    entities = [(entity.text, entity.label_)for entity in docx.ents]
    allData = ['"Token":{},\n"Entities":{}'.format(tokens, entities)]
    return allData

#Function for named entity recognition
def ner(my_text):
    nlp = spacy.load("xx_ent_wiki_sm")
    nlp.add_pipe("sensitizers")
    doc = nlp(my_text)
    html = displacy.render([doc], style="ent", page=False)
    #st.write(html, unsafe_allow_html=True)
    st.markdown(html, unsafe_allow_html=True)
    st.markdown(" <br> </br>", unsafe_allow_html= True)

    #displacy.serve(doc, style="ent")


def process_text(text):
    text = re.sub('[^A-Za-z]', ' ', text.lower())
    tokenized_text = word_tokenize(text)
    clean_text = [
        word for word in tokenized_text
        if word not in stopwords.words("xx_ent_wiki_sm")
    ]
    #gensim.parsing.stem_text(word)

    #word list only
    return clean_text




def topic_mod(my_text , num_topics=10,num_words=5):
    nlp = spacy.load('xx_ent_wiki_sm')
    nlp.add_pipe("sentencizer")
    doc = nlp(my_text)


    text_data = [sent.text.strip() for sent in doc.sents]
    #st.write(text_data)

    texts_lem = [process_text(text) for text in text_data]
    #st.write(texts)
    dictionary = gensim.corpora.Dictionary(texts_lem)
    #dictionary.filter_extremes(no_below=2, no_above=0.1)
    #dictionary.save_as_text ('dict.txt')
    corpus = [dictionary.doc2bow(text) for text in texts_lem]
    #temp = dictionary[0]  # This is only to "load" the dictionary.
    id2word = dictionary.id2token
    #corpora.MmCorpus.serialize ('cop.mm', corpus)
    #dictionary = gensim.corpora.Dictionary.load_from_text ('dict.txt')
    #corpus = corpora.MmCorpus ('cop.mm')
    model = gensim.models.LdaModel(corpus=corpus, num_topics=num_topics, id2word=id2word, passes=10,random_state =2)
    topics = model.print_topics(num_words=num_words)
    for topic in topics:
        st.write(topic)


    #lda_display = pyLDAvis.gensim.prepare(model, corpus, dictionary, sort_topics=True)
    #st.pyplot(pyLDAvis.display(lda_display))
    #pyLDAvis.display(lda_display)
    #plt.show()








def main():
    # Title
    front_up()
    st.title('Распознавание именованных объектов и тематическое моделирование')

    if st.checkbox('Показывать именованные объекты', key='ner'):
        st.subheader('Отобразить NER')
        boool, text = selection(key='ner')

        if st.button("Анализировать", key='ner'):
            if boool == 0:
                message = text
                ner(message)
                #st.json(nlp_result)

            else:
                try:
                    message = get_text(text)
                    ner(message)
                    #st.json(nlp_result)
                except BaseException as e:
                    st.warning(e)

    if st.checkbox('Показывать основные темы текста', key='topics'):
        st.subheader('Главные темы вашего текста')

        boool, text = selection(key='topics')
        num_topics=st.number_input('Номера тем',key='topic',step=1,min_value=1,format = '%d')
        num_words=st.number_input('Количество слов в теме' , key='words', format='%d', min_value=1,step=1)
        if st.button("Анализировать", key='topics'):
            if boool == 0:
                message = text
                #function here
                topic_mod(message,num_topics,num_words)
            else:
                try:
                    message = get_text(text)
                    #function here
                    topic_mod(message,num_topics,num_words)
                except BaseException as e:
                    st.warning(e)



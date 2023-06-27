import re
import streamlit as st
from pages.fetch import *
import sys
from gensim.summarization.summarizer import summarize
# Sumy Summary Pkg
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
import spacy
from summarizer import Summarizer as sz
import numpy as np
np.bool = np.bool

def sumy_summarizer(docx):
    parser = PlaintextParser.from_string(docx, Tokenizer("russian"))
    lex_summarizer = LexRankSummarizer()
    summary = lex_summarizer(parser.document, 3)
    summary_list = [str(sentence) for sentence in summary]
    result = ' '.join(summary_list)
    return result


def gensim_summarizer(my_text):
    nlp = spacy.load("ru_core_news_sm")
    doc = nlp(my_text)
    text_data = [sent.string.strip() for sent in doc.sents]
    return summarize(my_text)


def bert_sum(message):
    model = sz()
    result = model(message, min_length=60)
    full = ''.join(result)
    st.success(full)


def summarizevis(message, summary_options):
    if summary_options == 'sumy':
        st.text("Использование сумматора Summarizer ..")
        summary_result = sumy_summarizer(message)

    elif summary_options == 'gensim':
        st.text("Использование обобщителя Gensim ..")
        summary_result = gensim_summarizer(message)

    else:
        st.warning("Использование сумматора по умолчанию")
        st.text("Использование Gensim сумматора..")
        summary_result = sumy_summarizer(message)

    st.success(summary_result)


def main():
    # Title
    front_up()
    st.title('Краткое изложение текста')

    if st.checkbox('Краткое изложение извлеченного текста', key='ts'):
        st.subheader('Краткое изложение, основанное на извлечении')
        summary_options = st.selectbox("Выберите Обобщить", ['sumy', 'gensim'])
        boool, text = selection(key='ts')

        if st.button("Подвести итоги", key='ts'):
            if boool == 0:
                message = text
                summarizevis(message, summary_options)

            # st.json(nlp_result)

            else:
                try:
                    message = get_text(text)
                    summarizevis(message, summary_options)
                # st.json(nlp_result)
                except BaseException as e:
                    st.warning(e)

    if st.checkbox('Краткое изложение абстрактного текста', key='ats'):
 	    st.subheader('Абстрактная генерация')
 	    boool, text = selection(key='ats')

 	    if st.button("Подвести итоги", key='ats'):
 		    if boool == 0:
 			    message = text
 			    bert_sum(message)

 		    else:
 			    try:
 				    message = get_text(text)
 				    bert_sum(message)


 			    except BaseException as e:
 				    st.warning(e)

	

"""Основной модуль для приложения streamlit NLP"""
import streamlit as st


import pages.home
import pages.basicNLP
import pages.nertm
import pages.textSummarization
import pages.Recognition

PAGES = {
    "Home": pages.home,
    "Basic NLP": pages.basicNLP,
    "NER and Topic Modelling": pages.nertm,
    "Text Summarization": pages.textSummarization,
    "Recognition": pages.Recognition
}


def main():
    
    st.sidebar.title("NLP")
    st.sidebar.text("Обработка естественного языка")
    
    st.sidebar.title("Навигация")
    page = st.sidebar.radio("Переход", list(PAGES.keys()))

    #PAGES[page].main()


    with st.spinner(f"Загрузка {page} ..."):
        PAGES[page].main()

        
    
    st.sidebar.title("О приложении")
    
    st.sidebar.info(
        """
        Это приложение использует самые современные API бесплатного уровня с разных платформ
        например, IBM, Google Cloud и библиотеки, такие как Spacey, Genism, NLTK, Text blob и т.д. 
        Он использует Streamlit для реализации красивого и простого веб-приложения.
        """
    )

if __name__ == "__main__":
    main()

import streamlit as st

from bs4 import BeautifulSoup
from urllib.request import urlopen
# Fetch Text From Url

def get_text(raw_url):
	page = urlopen(raw_url)
	soup = BeautifulSoup(page)
	fetched_text = ' '.join(map(lambda p: p.text, soup.find_all('p')))
	return fetched_text

def selection(key):
	option = st.selectbox('Как бы вы хотели предоставить эти данные?',('URL', 'Вставить/Написать текст'), index=1, key=key)
	st.write('Вы выбрали:', option)
	if option == 'Вставить/Написать текст' :
		message = st.text_area("Введите текст", "Введите здесь ..", key=key+'text')
		return (0,message)
	else:
		url = st.text_area("Введите текст", "Введите здесь ..", key= key+'url')
		return (1,url)

def front_up():
    html_temp = """
		<div style="background-color:#ff1a75;padding:10px">
		<h1 style="color:white;text-align:center;">NLP</h1>
		<h4 style="color:white;text-align:center;">Обработка естественного языка...</h4>
		</div>
		<br></br>
		<br></br>
	"""
    st.markdown(html_temp,unsafe_allow_html=True)


def front_down():
    #closing remarks
    pass



def contact():
    pass    
	#st.markdown(html,unsafe_allow_html=True)

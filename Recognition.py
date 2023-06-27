import easyocr as ocr
import streamlit as st
from PIL import Image
import numpy as np
from pages.fetch import *
import numpy as np
import docx2txt
import pdfplumber
from PyPDF2 import PdfFileReader
import tweet as tweet
import xx_ent_wiki_sm
def main():
    front_up()
    st.title('Easy OCR - Extract Text from Images')
    st.markdown("## Optical Character Recognition - Using `easyocr`, `streamlit`")
    image = st.file_uploader(label="Upload your image here", type=['png', 'jpg', 'jpeg', 'img', 'tiff'])
    if image is not None:
        input_image = Image.open(image)  # read image
        st.image(input_image)
        with st.spinner("AI is at Work! "):

            result = reader.readtext(np.array(input_image))

            result_text = []  # empty list for results

            for text in result:
                result_text.append(text[1])

            st.write(result_text)
            # st.success("Here you go!")
        st.balloons()
    else:
        st.write("Upload an Image")


@st.cache
def load_model():
        reader = ocr.Reader(['en','ru','uk'], model_storage_directory='.')
        return reader
reader = load_model()


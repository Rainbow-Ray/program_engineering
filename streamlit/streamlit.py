from io import StringIO
from pathlib import Path
import streamlit as st
import importlib.util
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import main


st.title('Удаление фона на фото')
uploaded_file = st.file_uploader(label='Загрузите фотографию:', type=["jpg", "jpeg", "png"])
if st.button('Вырезать фон'):
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()



        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        res = main.io_file_input(stringio)
        st.image(res)


    a = 1
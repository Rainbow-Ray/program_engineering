from io import BytesIO
from io import StringIO
from pathlib import Path
import streamlit as st
import importlib.util
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from main import io_file_input


st.title('Удаление фона на фото')
uploaded_file = st.file_uploader(label='Загрузите фотографию:', type=["jpg", "jpeg", "png"])
if st.button('Вырезать фон'):
    if uploaded_file is not None:
        stringio = BytesIO(uploaded_file.getvalue())
        st.write(stringio)
        res = io_file_input(stringio)
        st.image(res)
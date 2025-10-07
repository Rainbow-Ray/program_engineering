from io import BytesIO
from io import StringIO
from pathlib import Path
import streamlit as st
import importlib.util
import sys
import numpy as np
from PIL import Image
sys.path.append(str(Path(__file__).resolve().parent.parent))
from main import preprocess_image
from main import postprocess_image
from main import model
from main import device
from main import io_file_input
# streamlit run streamlit\streamlit.py

def ass(img):
    if not hasattr(img, 'ndim'):
        return img

    if img.ndim > 2:
        if img.shape[-1] not in (3, 4) and img.shape[-3] in (3, 4):
            img = np.swapaxes(img, -1, -3)
            img = np.swapaxes(img, -2, -3)

    return img



st.title('Удаление фона на фото')
uploaded_file = st.file_uploader(label='Загрузите фотографию:', type=["jpg", "jpeg", "png"])
if st.button('Вырезать фон'):
    if uploaded_file is not None:

        bytes_data = uploaded_file.getvalue()
        image = Image.open(BytesIO(bytes_data))
        a = np.asarray(image)
        img = ass(a)
        res = io_file_input(img, bytes_data)
        # st.image(res)
        st.image(res)

        # bytes_data = uploaded_file.getvalue()
        # st.write(bytes_data)
        # To convert to a string based IO:


        # To read file as string:
        # string_data = stringio.read()
        # st.write(string_data)

        # Can be used wherever a "file-like" object is accepted:

        # res = io_file_input(stringio)
        # st.image(res)



from io import BytesIO
from io import StringIO
from pathlib import Path
import io
from PIL import Image

import streamlit as st
import importlib.util
import  numpy as np
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from main import io_file_input

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
        # array_from_bytes = np.frombuffer(bytes_data, dtype=np.uint8)
        # st.write(bytes_data)
        image = Image.open(io.BytesIO(bytes_data))
        a = np.asarray(image)
        img = ass(a)
        res = io_file_input(img, bytes_data)
        # st.image(res)


    a = 1
import streamlit as st

st.title('Удаление фона на фото')
st.file_uploader(label='Загрузите фотографию:', type=["jpg", "jpeg", "png"])
if st.button('Вырезать фон'):
    a =1
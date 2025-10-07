from io import BytesIO
from io import StringIO
from pathlib import Path
import streamlit as st
import importlib.util
import sys
import numpy as np
from PIL import Image
# streamlit run streamlit\streamlit.py
from PIL import Image
import io
from skimage import io as sio
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForImageSegmentation
from torchvision.transforms.functional import normalize


model = AutoModelForImageSegmentation.from_pretrained("briaai/RMBG-1.4",trust_remote_code=True)

def preprocess_image(im: np.ndarray, model_input_size: list) -> torch.Tensor:
    if len(im.shape) < 3:
        im = im[:, :, np.newaxis]
    im_tensor = torch.tensor(im, dtype=torch.float32).permute(2,0,1)
    im_tensor = F.interpolate(torch.unsqueeze(im_tensor,0), size=model_input_size, mode='bilinear')
    image = torch.divide(im_tensor,255.0)
    image = normalize(image,[0.5,0.5,0.5],[1.0,1.0,1.0])
    return image

def postprocess_image(result: torch.Tensor, im_size: list)-> np.ndarray:
    result = torch.squeeze(F.interpolate(result, size=im_size, mode='bilinear') ,0)
    ma = torch.max(result)
    mi = torch.min(result)
    result = (result-mi)/(ma-mi)
    im_array = (result*255).permute(1,2,0).cpu().data.numpy().astype(np.uint8)
    im_array = np.squeeze(im_array)
    return im_array

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

def show_image(path:str):
  img = Image.open(path)
  img.show()

#show_image("cat_walking.jpg")
#show_image("girafe.jpg")
#show_image("dog_fight.jpg")

def prepare_input (path:str):
  orig_im = sio.imread(path)
  orig_im_size = orig_im.shape[0:2]
  model_input_size = [1024, 1024]
  image = preprocess_image(orig_im, model_input_size).to(device)
  result=model(image)
  result_image = postprocess_image(result[0][0], orig_im_size)
  pil_mask_im = Image.fromarray(result_image)
  orig_image = Image.open(path)
  no_bg_image = orig_image.copy()
  no_bg_image.putalpha(pil_mask_im)
  no_bg_image.show()

# def io_file_input(file_object):
#     orig_image = file_object
#     orig_im = np.array(orig_image)
#     orig_im_size = orig_im.shape[0:2]
#     model_input_size = [1024, 1024]
#     image = preprocess_image(orig_im, model_input_size).to(device)
#     result = model(image)
#     result_image = postprocess_image(result[0][0], orig_im_size)
#     pil_mask_im = Image.fromarray(result_image)
#     orig_image = Image.open(file_object)
#     no_bg_image = orig_image.copy()
#     no_bg_image.putalpha(pil_mask_im)
#     return no_bg_image

def io_file_input(orig_im, bytes_data):
    orig_im_size = orig_im.shape[0:2]
    model_input_size = [1024, 1024]
    image = preprocess_image(orig_im, model_input_size).to(device)
    result = model(image)
    result_image = postprocess_image(result[0][0], orig_im_size)
    pil_mask_im = Image.fromarray(result_image)
    orig_image = Image.open(io.BytesIO(bytes_data))
    no_bg_image = orig_image.copy()
    no_bg_image.putalpha(pil_mask_im)
    return no_bg_image


def ass(img):
    if not hasattr(img, 'ndim'):
        return img

    if img.ndim > 2:
        if img.shape[-1] not in (3, 4) and img.shape[-3] in (3, 4):
            img = np.swapaxes(img, -1, -3)
            img = np.swapaxes(img, -2, -3)

    return img



st.title('Удаление фона на фото')
resimg = st.empty

def delim():
    st.empty

uploaded_file = st.file_uploader(label='Загрузите фотографию:', type=["jpg", "jpeg", "png"])

if st.button('Вырезать фон'):
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        image = Image.open(BytesIO(bytes_data))
        a = np.asarray(image)
        img = ass(a)
        res = io_file_input(img, bytes_data)
        # st.image(res)
        # bytes_data = uploaded_file.getvalue()
        # st.write(bytes_data)
        # To convert to a string based IO:
        with st.empty():
            st.image(res)

        # To read file as string:
        # string_data = stringio.read()
        # st.write(string_data)

        # Can be used wherever a "file-like" object is accepted:

        # res = io_file_input(stringio)
        # st.image(res)


uploaded_file = None
res = None
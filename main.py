from PIL import Image
from skimage import io
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
  orig_im = io.imread(path)
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

def io_file_input(file_object):
    orig_image = Image.open(file_object)
    orig_im = np.array(orig_image)
    orig_im_size = orig_im.shape[0:2]
    model_input_size = [1024, 1024]
    image = preprocess_image(orig_im, model_input_size).to(device)
    result = model(image)
    result_image = postprocess_image(result[0][0], orig_im_size)
    pil_mask_im = Image.fromarray(result_image)
    orig_image = Image.open(file_object)
    no_bg_image = orig_image.copy()
    no_bg_image.putalpha(pil_mask_im)
    return no_bg_image

import streamlit as st
from io import BytesIO

st.title('Удаление фона на фото')
uploaded_file = st.file_uploader(label='Загрузите фотографию:', type=["jpg", "jpeg", "png"])
if st.button('Вырезать фон'):
    if uploaded_file is not None:

        stringio = BytesIO(uploaded_file.getvalue())
        st.write(stringio)
        res = io_file_input(stringio)
        st.image(res)
import streamlit as st
from model import LocalizationModel
from streamlit_drawable_canvas import st_canvas
import cv2
import pandas as pd
import torchvision.transforms as T
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt



st.title('Digit Recognizer App')


device='cpu'
model = LocalizationModel()
model.to(device)
model.load_state_dict(torch.load('weights.pt', map_location=torch.device('cpu')))

st.subheader('Draw a digit!')

st.sidebar.header("Settings")
stroke_width = st.sidebar.slider("Brush width: ", 10, 30, 20)

transform = T.Compose([
    T.ToTensor(),
])


SIZE = 300
canvas_result = st_canvas(
    fill_color='#000000',
    stroke_width=stroke_width,
    stroke_color='#FFFFFF',
    background_color='#000000',
    width=SIZE,
    height=SIZE,
    drawing_mode="freedraw",
    key='canvas')


if canvas_result.image_data is not None:

    img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
    rescaled = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST) 
    #rescaled = transform(rescaled)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    


    clicked = st.button('Run')
    #st.subheader('Image that went through model')
    if clicked:
        
            with st.spinner(text='In progress'):
                new_size = (100, 100)
                image = T.ToPILImage()(img_gray)
                original_size = image.size
                left_pad = np.random.randint(0, new_size[0] - original_size[0])
                top_pad = np.random.randint(0, new_size[1] - original_size[1])
                right_pad = new_size[0] - original_size[0] - left_pad
                bottom_pad = new_size[1] - original_size[1] - top_pad
                padded_img = ImageOps.expand(image, (left_pad, top_pad, right_pad, bottom_pad), fill=0)
                #st.image(padded_img, width=300)
                transformed_image = transform(padded_img)
                model.eval()
                with torch.no_grad():
                     logit, coords = model(transformed_image.unsqueeze(0))
                     
                fig, ax = plt.subplots(1, 1, squeeze=True)
                img = cv2.cvtColor(np.array(padded_img), cv2.COLOR_BGR2RGB)

                pred_box_coords = (coords.cpu().numpy() * 100).astype(int)
                pic_pred = cv2.rectangle(
                    img.copy(),
                    (pred_box_coords[0][0], pred_box_coords[0][1]),
                    (pred_box_coords[0][2], pred_box_coords[0][3]),
                    color=(0, 0, 255), thickness=1
                    )
                plt.imshow(pic_pred) # рисуем все вместе
                ax.set_xticks([]); ax.set_yticks([]) # убираем тики
                plt.savefig('predict.png')
                st.subheader(f'Predicted digit: {logit.argmax(1).item()}')
                st.image('predict.png')
            st.info('Run Successful !')
            #st.write(logit.argmax(1).item(), coords)
            
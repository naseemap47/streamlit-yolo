import streamlit as st
import cv2
from utils.hubconf import custom
import numpy as np


st.title('YOLOv7 Predictions')
sample_img = cv2.imread('sample.jpg')
sample_img = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)
FRAME_WINDOW = st.image(sample_img)
pred = st.checkbox('Predict Using YOLOv7')
st.sidebar.title('Settings')
st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child(width: 400px;)
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child(width: 400px; margin-left: -400px)
    </style>
    """,
unsafe_allow_html=True
)
st.sidebar.markdown('---')
st.sidebar.text('Options')
webcam = st.sidebar.checkbox('Webcam')
st.sidebar.markdown('---')
confidence = st.sidebar.slider('Confidence', min_value=0.0, max_value=1.0, value=0.25)
st.sidebar.markdown('---')
upload_img_file = st.sidebar.file_uploader('Upload Image', type=['jpg', 'jpeg', 'png'])
print(upload_img_file)
if upload_img_file is not None:
    file_bytes = np.asarray(bytearray(upload_img_file.read()), dtype=np.uint8)
    opencv_img = cv2.imdecode(file_bytes, 1)
    FRAME_WINDOW = st.image(opencv_img, channels='BGR')

if pred:
    model = custom(path_or_model='yolov7.pt')

if webcam:
    cap = cv2.VideoCapture(0)
    while True:
        success, img = cap.read()
        img_rgb = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        FRAME_WINDOW.image(img_rgb)
        
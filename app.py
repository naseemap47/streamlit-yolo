import streamlit as st
import cv2
from utils.hubconf import custom
from utils.plots import plot_one_box
import numpy as np
import tempfile


st.title('YOLOv7 Predictions')
sample_img = cv2.imread('sample.jpg')
FRAME_WINDOW = st.image(sample_img, channels='BGR')
st.sidebar.title('Settings')

options = st.sidebar.radio('Options', ('Webcam','Image','Video'),index=1)

confidence = st.sidebar.slider('Detection Confidence', min_value=0.0, max_value=1.0, value=0.25)

# Image
if options=='Image':
    pred = st.checkbox('Predict Using YOLOv7')
    upload_img_file = st.sidebar.file_uploader('Upload Image', type=['jpg', 'jpeg', 'png'])

    if upload_img_file is not None:
        file_bytes = np.asarray(bytearray(upload_img_file.read()), dtype=np.uint8)
        opencv_img = cv2.imdecode(file_bytes, 1)
        FRAME_WINDOW.image(opencv_img, channels='BGR')

        if pred:
            model = custom(path_or_model='yolov7.pt')
            bbox_list = []
            results = model(opencv_img)
            # Bounding Box
            box = results.pandas().xyxy[0]
            class_list = box['class'].to_list()
            f = open('class.txt', 'r').read()
            class_labels = f.split("\n")
            for i in box.index:
                xmin, ymin, xmax, ymax, conf = int(box['xmin'][i]), int(box['ymin'][i]), int(box['xmax'][i]), \
                    int(box['ymax'][i]), box['confidence'][i]
                if conf > confidence:
                    bbox_list.append([xmin, ymin, xmax, ymax])
            if len(bbox_list)!=0:
                for bbox, id in zip(bbox_list, class_list):
                    plot_one_box(bbox, opencv_img, label=class_labels[id], color=[0,0,255], line_thickness=2)
                FRAME_WINDOW.image(opencv_img, channels='BGR')

# Video
if options=='Video':
    pred = st.checkbox('Predict Using YOLOv7')
    upload_video_file = st.sidebar.file_uploader('Upload Video', type=['mp4', 'avi', 'mkv'])
    if upload_video_file is not None:
        model = custom(path_or_model='yolov7.pt')
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(upload_video_file.read())
        cap = cv2.VideoCapture(tfile.name)
        success, img = cap.read()
        if pred:
            while success:
                bbox_list = []
                results = model(img)
                # Bounding Box
                box = results.pandas().xyxy[0]
                class_list = box['class'].to_list()
                f = open('class.txt', 'r').read()
                class_labels = f.split("\n")
                for i in box.index:
                    xmin, ymin, xmax, ymax, conf = int(box['xmin'][i]), int(box['ymin'][i]), int(box['xmax'][i]), \
                        int(box['ymax'][i]), box['confidence'][i]
                    if conf > confidence:
                        bbox_list.append([xmin, ymin, xmax, ymax])
                if len(bbox_list)!=0:
                    for bbox, id in zip(bbox_list, class_list):
                        plot_one_box(bbox, img, label=class_labels[id], color=[0,0,255], line_thickness=2)
                    FRAME_WINDOW.image(img, channels='BGR')
                else:
                    FRAME_WINDOW = st.image([])
                    print('[INFO] Failied to Loadd Video')
                    break
    else:
        video_file = open('test.mp4', 'rb')
        video_bytes = video_file.read()
        FRAME_WINDOW.video(video_bytes)
# Web-cam
if options=='Webcam':
    cap = cv2.VideoCapture(0)
    while True:
        success, img = cap.read()
        img_rgb = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        FRAME_WINDOW.image(img_rgb)
        
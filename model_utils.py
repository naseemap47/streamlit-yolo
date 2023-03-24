from utils.plots import plot_one_box
from PIL import ImageColor
import subprocess
import streamlit as st
import psutil
import pandas as pd


def get_gpu_memory():
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    return gpu_memory[0]

def color_picker_fn(classname, key):
    color_picke = st.sidebar.color_picker(f'{classname}:', '#ff0003', key=key)
    color_rgb_list = list(ImageColor.getcolor(str(color_picke), "RGB"))
    color = [color_rgb_list[2], color_rgb_list[1], color_rgb_list[0]]
    return color


def get_yolo(img, model_type, model, confidence, color_pick_list, class_list, draw_thick):
    current_no_class = []
    results = model(img)
    if model_type == 'YOLOv5':
        for result in results:
            bboxs = result.boxes.xyxy
            conf = result.boxes.conf
            cls = result.boxes.cls
            for bbox, cnf, cs in zip(bboxs, conf, cls):
                xmin = int(bbox[0])
                ymin = int(bbox[1])
                xmax = int(bbox[2])
                ymax = int(bbox[3])
                if cnf > confidence:
                    plot_one_box([xmin, ymin, xmax, ymax], img, label=class_list[int(cs)],
                                    color=color_pick_list[int(cs)], line_thickness=draw_thick, confi=cnf)
                    current_no_class.append([class_list[int(cs)]])

    if model_type == 'YOLOv8':
        for result in results:
            bboxs = result.boxes.xyxy
            conf = result.boxes.conf
            cls = result.boxes.cls
            for bbox, cnf, cs in zip(bboxs, conf, cls):
                xmin = int(bbox[0])
                ymin = int(bbox[1])
                xmax = int(bbox[2])
                ymax = int(bbox[3])
                if cnf > confidence:
                    plot_one_box([xmin, ymin, xmax, ymax], img, label=class_list[int(cs)],
                                    color=color_pick_list[int(cs)], line_thickness=draw_thick, confi=cnf)
                    current_no_class.append([class_list[int(cs)]])
    return img, current_no_class


def get_system_stat(stframe1, stframe2, stframe3, fps, df_fq):
    # Updating Inference results
    with stframe1.container():
        st.markdown("<h2>Inference Statistics</h2>", unsafe_allow_html=True)
        if round(fps, 4)>1:
            st.markdown(f"<h4 style='color:green;'>Frame Rate: {round(fps, 4)}</h4>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h4 style='color:red;'>Frame Rate: {round(fps, 4)}</h4>", unsafe_allow_html=True)
    
    with stframe2.container():
        st.markdown("<h3>Detected objects in curret Frame</h3>", unsafe_allow_html=True)
        st.dataframe(df_fq, use_container_width=True)

    with stframe3.container():
        st.markdown("<h2>System Statistics</h2>", unsafe_allow_html=True)
        js1, js2, js3 = st.columns(3)                       

        # Updating System stats
        with js1:
            st.markdown("<h4>Memory usage</h4>", unsafe_allow_html=True)
            mem_use = psutil.virtual_memory()[2]
            if mem_use > 50:
                js1_text = st.markdown(f"<h5 style='color:red;'>{mem_use}%</h5>", unsafe_allow_html=True)
            else:
                js1_text = st.markdown(f"<h5 style='color:green;'>{mem_use}%</h5>", unsafe_allow_html=True)

        with js2:
            st.markdown("<h4>CPU Usage</h4>", unsafe_allow_html=True)
            cpu_use = psutil.cpu_percent()
            if mem_use > 50:
                js2_text = st.markdown(f"<h5 style='color:red;'>{cpu_use}%</h5>", unsafe_allow_html=True)
            else:
                js2_text = st.markdown(f"<h5 style='color:green;'>{cpu_use}%</h5>", unsafe_allow_html=True)

        with js3:
            st.markdown("<h4>GPU Memory Usage</h4>", unsafe_allow_html=True)  
            try:
                js3_text = st.markdown(f'<h5>{get_gpu_memory()} MB</h5>', unsafe_allow_html=True)
            except:
                js3_text = st.markdown('<h5>NA</h5>', unsafe_allow_html=True)

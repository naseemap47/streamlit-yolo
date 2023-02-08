import streamlit as st
import cv2
import torch
from utils.hubconf import custom
from utils.plots import plot_one_box
import numpy as np
import tempfile
from PIL import ImageColor
import time
from collections import Counter
import json
import psutil
import subprocess
import pandas as pd
from ultralytics import YOLO


def get_yolo(img, model_type, model, confidence, color_pick_list, class_list, draw_thick):
    current_no_class = []
    results = model(img)
    if model_type == 'YOLOv7':
        box = results.pandas().xyxy[0]

        for i in box.index:
            xmin, ymin, xmax, ymax, conf, id, class_name = int(box['xmin'][i]), int(box['ymin'][i]), int(box['xmax'][i]), \
                int(box['ymax'][i]), box['confidence'][i], box['class'][i], box['name'][i]
            if conf > confidence:
                plot_one_box([xmin, ymin, xmax, ymax], img, label=class_name,
                                color=color_pick_list[id], line_thickness=draw_thick)
            current_no_class.append([class_name])

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
                                    color=color_pick_list[int(cs)], line_thickness=draw_thick)
                    current_no_class.append([class_list[int(cs)]])
    return img, current_no_class

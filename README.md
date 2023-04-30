# YOLO Streamlit Dashbord
Display predicted Video, Images and webcam using YOLO models (**YOLOv7** & **YOLOv8**) with Streamlit

### Sample Streamlit YOLO Dashboard
Streamlit Dashboard: https://naseemap47-streamlit-yolo-app-v7gbfg.streamlit.app/

## Docker
dockerhub: https://hub.docker.com/repository/docker/naseemap47/streamlit-yolo

#### 1. Pull Docker Image
```
docker pull naseemap47/streamlit-yolo
```
#### 2. Change permistion
```
sudo xhost +si:localuser:root
```
#### 3. RUN Docker Image
```
docker run --gpus all -e DISPLAY=$DISPLAY -v /tmp/.X11-unix/:/tmp/.X11-unix --ipc=host --device=/dev/video0:/dev/video0 -p 8502 -it --rm naseemap47/streamlit-yolo
```

## 🚀 New Update (10/02/2023)
Integrated new YOLOv8 model, now you can run YOLOv8 model on RTSP, Webcam, Video and Image

## Streamlit Options
### Modes
 - RTSP
 - Webcam
 - Video
 - Image
 
 ## Sample Streamlit Dashboard Output
 
 [out.webm](https://user-images.githubusercontent.com/88816150/193816239-b351c3d6-1d9a-4820-87b5-0cfec1ad5d90.webm)

 ## StepUp
```
git clone https://github.com/naseemap47/streamlit-yolo.git
cd streamlit-yolo
```
Install dependency
```
pip3 install -r requirements.txt
```
Run **Streamlit**
```
streamlit run app.py
```

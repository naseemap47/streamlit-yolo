# Streamlit YOLOv7
<p align="center">
 <img src="https://github.com/naseemap47/streamlit-yolov7/blob/master/sample.jpg">
</p>
Display predicted Video, Images and webcam using YOLOv7 model with Streamlit

Streamlit Dashboard: https://naseemap47-streamlit-yolov7-app-deploy-bfr4xt.streamlitapp.com/

Streamlit Server don't have GPU, so to deploy model, I used `deploy` branch (without **GPU** option)

Only `master` contain all codes and details. (both **GPU** and **CPU** options)

So better to clone `master` branch and run in your own system

## Streamlit Options
 - RTSP
 - Webcam
 - Video
 - Image
 ## StepUp
```
git clone https://github.com/naseemap47/streamlit-yolov7.git
cd streamlit-yolov7
```
Install dependency
```
pip3 install -r requirements.txt
```
Run **Streamlit**
```
streamlit run app.py
```

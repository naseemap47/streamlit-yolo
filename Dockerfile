FROM ubuntu:20.04
ARG DEBIAN_FRONTEND=noninteractive
COPY . /App
RUN apt-get update && \
apt-get install -y \
python3 \
python3-pip \
ffmpeg \
libsm6 \
libxext6 \
git
WORKDIR /App
RUN cd streamlit-yolo
RUN pip install -r requirements.txt
CMD [ "streamlit", "run", "app.py" ]
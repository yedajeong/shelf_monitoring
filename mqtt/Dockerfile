FROM ubuntu:latest

RUN apt-get update
RUN apt-get upgrade -y python3
RUN apt install -y python3-pip

COPY requirements.txt .

RUN pip install -r requirements.txt
RUN apt-get update
RUN apt-get -y install libgl1-mesa-glx
RUN apt-get install -y libglib2.0-0

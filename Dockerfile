FROM ubuntu:latest

RUN apt update

RUN apt install -y python3  python3-pip vim 

WORKDIR /app

COPY requirements.txt .

COPY  . /app/

RUN pip install --no-cache-dir --break-system-packages -r requirements.txt
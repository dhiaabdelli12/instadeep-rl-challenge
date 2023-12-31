FROM python:3.8

WORKDIR /app

COPY . /app

RUN apt-get update && apt-get install -y swig

RUN make install

VOLUME /app/checkpoints


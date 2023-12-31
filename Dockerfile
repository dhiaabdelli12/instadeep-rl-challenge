FROM python:3.8

WORKDIR /app

COPY . /app

RUN make install

VOLUME /app/checkpoints


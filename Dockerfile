FROM nvidia/cuda:12.6.2-runtime-ubuntu22.04

WORKDIR /app
COPY . /app

RUN apt-get update && \
    add-apt-repository universe \
    apt update \
    apt install python3.12 \
    rm -rf /var/lib/apt/lists/*

RUN /usr/local/bin/pip3.12 install --no-cache-dir .

EXPOSE 8000

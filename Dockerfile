FROM nvidia/cuda:12.6.2-runtime-ubuntu22.04

WORKDIR /app
COPY . /app

RUN apt-get update && \
    apt-get install -y python3.12 python3.12-venv python3.12-dev python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN pip install .

EXPOSE 8000
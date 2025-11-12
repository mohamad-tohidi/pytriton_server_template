FROM nvidia/cuda:12.6.2-runtime-ubuntu22.04

WORKDIR /app
COPY . /app

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && \
    apt-get install -y python3.12 python3.12-venv python3.12-distutils && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 && \
    rm -rf /var/lib/apt/lists/*

RUN pip install .

EXPOSE 8000
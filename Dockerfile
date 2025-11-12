FROM nvidia/cuda:12.6.2-runtime-ubuntu22.04

WORKDIR /app
COPY . /app

RUN apt-get update && \
    apt install -y python3.12 && \
    rm -rf /var/lib/apt/lists/*

RUN python3.12 --version && pip --version

RUN pip install --no-cache-dir .

EXPOSE 8000


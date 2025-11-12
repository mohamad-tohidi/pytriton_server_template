FROM nvidia/cuda:12.6.2-runtime-ubuntu22.04

WORKDIR /app
COPY . /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      software-properties-common \
      curl \
      ca-certificates \
      build-essential \
      libssl-dev \
      libffi-dev \
      libbz2-dev \
      libreadline-dev \
      zlib1g-dev && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
      python3.12 \
      python3.12-venv \
      python3.12-dev && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 && \
    rm -rf /var/lib/apt/lists/*

RUN /usr/local/bin/pip3.12 install --no-cache-dir .

EXPOSE 8000

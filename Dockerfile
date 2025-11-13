FROM nvidia/cuda:12.6.2-runtime-ubuntu22.04 



ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ca-certificates \
      software-properties-common \
      wget \
      gnupg \
      lsb-release \
 && add-apt-repository ppa:deadsnakes/ppa -y \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
      python3.12 \
      python3.12-venv \
      python3.12-dev \
      python3.12-distutils \
      build-essential \
 && rm -rf /var/lib/apt/lists/*

RUN wget -q https://bootstrap.pypa.io/get-pip.py -O /tmp/get-pip.py \
 && python3.12 /tmp/get-pip.py \
 && rm /tmp/get-pip.py

 WORKDIR /app

COPY . .
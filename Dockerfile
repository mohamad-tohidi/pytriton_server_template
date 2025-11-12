FROM nvidia/cuda:12.6.2-runtime-ubuntu22.04
WORKDIR /app
COPY . /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends software-properties-common curl ca-certificates && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends python3.12 python3.12-venv python3.12-dev python3.12-distutils build-essential && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 && \
    rm -rf /var/lib/apt/lists/*

RUN /usr/local/bin/pip3.12 install --no-cache-dir .

EXPOSE 8000

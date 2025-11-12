FROM nvidia/cuda:12.6.2-runtime-ubuntu22.04

WORKDIR /app
COPY . /app

# Install Python 3.12 from source in a single efficient layer
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential zlib1g-dev libncurses5-dev libgdbm-dev \
        libnss3-dev libssl-dev libreadline-dev libffi-dev \
        libsqlite3-dev libbz2-dev wget ca-certificates curl && \
    wget -q https://www.python.org/ftp/python/3.12.7/Python-3.12.7.tgz && \
    tar -xf Python-3.12.7.tgz && \
    cd Python-3.12.7 && \
    ./configure --enable-optimizations --enable-shared && \
    make -j$(nproc) && \
    make install && \
    ldconfig && \
    cd .. && \
    rm -rf Python-3.12.7 Python-3.12.7.tgz && \
    apt-get remove -y --purge build-essential && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /root/.cache

# Install pip
RUN python3.12 -m ensurepip --upgrade && \
    python3.12 -m pip install --no-cache-dir pip setuptools wheel

EXPOSE 8000
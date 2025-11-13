FROM nvidia/cuda:12.6.2-runtime-ubuntu22.04 



ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Tehran

RUN ln -fs /usr/share/zoneinfo/$TZ /etc/localtime && \
    echo $TZ > /etc/timezone && \
    apt update && \
    apt install -y software-properties-common tzdata && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt update && apt install -y python3.12 python3.12-venv && \
    rm -rf /var/lib/apt/lists/*


WORKDIR /app

COPY . .
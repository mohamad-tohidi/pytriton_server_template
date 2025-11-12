FROM nvidia/cuda:12.6.2-runtime-ubuntu22.04

WORKDIR /app
COPY . /app

RUN apt-get update && apt-get install -y python3 python3-pip
RUN pip install uv && uv sync --frozen --no-dev 

EXPOSE 8000

FROM python:3.12-slim as builder
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir .

FROM nvidia/cuda:12.6.2-runtime-ubuntu22.04
ENV PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y python3.12 && rm -rf /var/lib/apt/lists/*
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH
WORKDIR /app

EXPOSE 8000
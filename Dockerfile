FROM nvidia/cuda:12.0.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3.10 python3.10-venv

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

COPY . /app

RUN uv sync --frozen --no-dev --extra cu128

EXPOSE 8000
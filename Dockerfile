FROM ghcr.io/astral-sh/uv:python3.10-bookworm-slim

WORKDIR /app

COPY . /app

RUN uv sync --frozen --no-dev --extra cu128

EXPOSE 8000
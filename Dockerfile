FROM nvidia/cuda:12.0.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3.10 python3-pip curl
RUN curl -sSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

WORKDIR /app
COPY . /app

RUN uv sync --frozen --no-dev

EXPOSE 8080

CMD ["uv", "run", "uvicorn", "pytriton_template.api:app", "--host", "0.0.0.0", "--port", "8080"]
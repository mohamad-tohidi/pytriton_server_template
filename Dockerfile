FROM nvcr.io/nvidia/tritonserver:24.04-py3

RUN pip install uv

WORKDIR /app
COPY . /app

RUN uv sync --frozen --no-dev 

EXPOSE 8000
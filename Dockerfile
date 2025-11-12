FROM python:3.12-slim 
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir .

FROM nvidia/cuda:12.6.2-runtime-ubuntu22.04
ENV PYTHONUNBUFFERED=1

COPY --from=python:3.12-slim  /install /usr/local

ENV PATH=/root/.local/bin:$PATH
WORKDIR /app

EXPOSE 8000
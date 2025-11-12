FROM python:3.12-slim AS python-base
FROM nvidia/cuda:12.6.2-runtime-ubuntu22.04

COPY --from=python-base /usr/local /usr/local
ENV PATH="/usr/local/bin:${PATH}"

WORKDIR /app
COPY . /app

RUN pip3 install --no-cache-dir .

EXPOSE 8000

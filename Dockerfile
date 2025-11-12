FROM tritonserver:25.02-pt

RUN pip install uv

WORKDIR /app
COPY . /app

RUN uv sync --frozen --no-dev 

EXPOSE 8000
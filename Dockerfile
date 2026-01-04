FROM mohamadtohidi/pytriton:latest

WORKDIR /src

COPY . .

RUN pip install .

EXPOSE 8000

RUN chmod +x /src/entrypoint.sh

ENTRYPOINT ["/src/entrypoint.sh"]  
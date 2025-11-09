# the simple fastAPI wrapper

import numpy as np
from fastapi import FastAPI, Request
from pydantic import BaseModel
from contextlib import asynccontextmanager
from pytriton.client import AsyncioModelClient


class EmbedRequest(BaseModel):
    text: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    client = AsyncioModelClient(url="grpc://localhost:8012", model_name="bge-m3")
    yield {"client": client}
    await client.close()


app = FastAPI(lifespan=lifespan)


@app.post("/embed")
async def get_embedding(req: EmbedRequest, request: Request):
    client: AsyncioModelClient = request.app.state["client"]
    # Prepare input: shape (1,), dtype object with bytes
    texts = np.array([req.text.encode("utf-8")], dtype=object)
    result = await client.infer_sample(texts=texts)
    embeddings = result["embeddings"]
    return embeddings.tolist()

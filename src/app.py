# the simple fastAPI wrapper
import numpy as np
import json
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from pytriton.client import (
    AsyncioModelClient,
)
from typing import List, Dict, Any
from enum import Enum

# --- Configuration ---
TRITON_GRPC_URL = "grpc://localhost:8012"
MODEL_NAME = "bge-m3"


class EmbeddingType(str, Enum):
    dense = "dense"
    lexical = "lexical"
    colbert = "colbert"
    all = "all"


class EmbedRequest(BaseModel):
    texts: List[str]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage the Triton client lifecycle.
    The client is created on startup and closed on shutdown.
    """
    client = AsyncioModelClient(
        url=TRITON_GRPC_URL, model_name=MODEL_NAME
    )
    print(f"Connecting to Triton at {TRITON_GRPC_URL}...")
    yield {"client": client}
    print("Closing Triton client connection...")
    await client.close()


app = FastAPI(
    title="BGE-M3 Embedding Wrapper",
    description="A FastAPI wrapper for the BGE-M3 PyTriton server.",
    lifespan=lifespan,
)


async def call_triton_server(
    client: AsyncioModelClient, texts: List[str]
) -> Dict[str, np.ndarray]:
    """
    Sends a batch of texts to the Triton server and returns the raw results.
    """
    input_texts_np = np.array(texts, dtype=np.bytes_)

    try:
        results_dict = await client.infer_batch(
            texts=input_texts_np
        )
        return results_dict
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred: {e}",
        )


# --- API Endpoints ---


@app.post("/bge-m3/{type}", response_model=Dict[str, Any])
async def get_embedding(
    type: EmbeddingType,
    body: EmbedRequest,
    request: Request,
):
    """
    Get embeddings from the BGE-M3 model.

    - **type**: The type of embedding to return:
        - `dense`: (BatchSize, 1024) float vectors
        - `lexical`: (BatchSize) list of sparse vector dictionaries
        - `colbert`: (BatchSize) list of ragged (Tokens, 1024) float vectors
        - `all`: All three types
    """

    client = request.state.client

    triton_results = await call_triton_server(
        client, body.texts
    )

    output = {}

    if (
        type == EmbeddingType.dense
        or type == EmbeddingType.all
    ):
        dense_vecs = triton_results["dense_vecs"]
        output["dense_vecs"] = dense_vecs.tolist()

    if (
        type == EmbeddingType.lexical
        or type == EmbeddingType.all
    ):
        lexical_bytes = triton_results["lexical_weights"]
        output["lexical_weights"] = [
            json.loads(b.decode("utf-8"))
            for b in lexical_bytes
        ]

    if (
        type == EmbeddingType.colbert
        or type == EmbeddingType.all
    ):
        colbert_bytes = triton_results["colbert_vecs"]
        output["colbert_vecs"] = [
            json.loads(b.decode("utf-8"))
            for b in colbert_bytes
        ]

    return output


@app.get("/")
def read_root():
    return {
        "message": "BGE-M3 FastAPI wrapper is running.",
        "endpoints": [
            "POST /bge-m3/dense",
            "POST /bge-m3/lexical",
            "POST /bge-m3/colbert",
            "POST /bge-m3/all",
        ],
    }


if __name__ == "__main__":
    print(
        "Starting FastAPI server on http://127.0.0.1:8000"
    )
    uvicorn.run(app, host="127.0.0.1", port=8000)

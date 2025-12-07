# mock_app.py - Mock version of the FastAPI wrapper without Triton calls

import numpy as np
import orjson as json
import uvicorn
from fastapi import FastAPI, Request
from pydantic import BaseModel
from contextlib import asynccontextmanager

from typing import List, Dict, Any
from enum import Enum

# --- Configuration ---
# No Triton URL or model name needed


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
    No Triton client in mock version.
    """
    yield {}


app = FastAPI(
    title="Mock BGE-M3 Embedding Wrapper",
    description="A mock FastAPI wrapper for BGE-M3 without Triton server calls.",
    lifespan=lifespan,
)


async def call_triton_server(
    texts: List[str],
) -> Dict[str, np.ndarray]:
    # Simulate input preparation
    if not texts:
        input_texts_np = np.empty((0, 1), dtype=object)
    else:
        encoded = [text.encode("utf-8") for text in texts]
        input_texts_np = np.array(encoded, dtype=object)[
            :, np.newaxis
        ]

    # Generate mock results
    n = len(texts)
    results_dict = {}

    # Mock dense_vecs: (N, 1024)
    results_dict["dense_vecs"] = np.zeros(
        (n, 1024), dtype=np.float32
    )

    # Mock lexical_weights: (N, 1) with bytes of json dict
    lexical_data = [
        {} for _ in range(n)
    ]  # empty dicts for simplicity
    lexical_json = [json.dumps(d) for d in lexical_data]
    lexical_bytes = [
        b for b in lexical_json
    ]  # already bytes from orjson
    lexical_np = np.array(lexical_bytes, dtype=object)[
        :, np.newaxis
    ]
    results_dict["lexical_weights"] = lexical_np

    # Mock colbert_vecs: (N, 1) with bytes of json list of lists
    colbert_data = [
        [np.zeros(1024, dtype=np.float32).tolist()]
        for _ in range(n)
    ]  # one zero vector per text
    colbert_json = [json.dumps(d) for d in colbert_data]
    colbert_bytes = [
        b for b in colbert_json
    ]  # already bytes from orjson
    colbert_np = np.array(colbert_bytes, dtype=object)[
        :, np.newaxis
    ]
    results_dict["colbert_vecs"] = colbert_np

    return results_dict


async def process_embedding_request(
    texts: List[str],
    type: EmbeddingType,
) -> Dict[str, Any]:
    """
    Refactored core logic to call mock Triton and process results based on type.
    """
    triton_results = await call_triton_server(texts)

    output = {}

    if (
        type == EmbeddingType.dense
        or type == EmbeddingType.all
    ):
        # This is correct, shape is (N, 1024)
        dense_vecs = triton_results["dense_vecs"]
        output["dense_vecs"] = dense_vecs.tolist()

    if (
        type == EmbeddingType.lexical
        or type == EmbeddingType.all
    ):
        # This now has shape (N, 1)
        lexical_bytes = triton_results["lexical_weights"]
        output["lexical_weights"] = [
            json.loads(b[0]) for b in lexical_bytes
        ]

    if (
        type == EmbeddingType.colbert
        or type == EmbeddingType.all
    ):
        # This now has shape (N, 1)
        colbert_bytes = triton_results["colbert_vecs"]
        output["colbert_vecs"] = [
            json.loads(b[0]) for b in colbert_bytes
        ]

    return output


@app.post(
    "/bge-m3/dense",
    response_model=Dict[str, Any],
    tags=["Embeddings"],
)
async def get_dense_embedding(
    body: EmbedRequest,
    request: Request,
):
    """
    Get dense embeddings from the mock BGE-M3 model.
    (BatchSize, 1024) float vectors
    """
    return await process_embedding_request(
        texts=body.texts,
        type=EmbeddingType.dense,
    )


@app.post(
    "/bge-m3/lexical",
    response_model=Dict[str, Any],
    tags=["Embeddings"],
)
async def get_lexical_embedding(
    body: EmbedRequest,
    request: Request,
):
    """
    Get lexical (sparse) embeddings from the mock BGE-M3 model.
    (BatchSize) list of sparse vector dictionaries
    """
    return await process_embedding_request(
        texts=body.texts,
        type=EmbeddingType.lexical,
    )


@app.post(
    "/bge-m3/colbert",
    response_model=Dict[str, Any],
    tags=["Embeddings"],
)
async def get_colbert_embedding(
    body: EmbedRequest,
    request: Request,
):
    """
    Get colbert (ragged) embeddings from the mock BGE-M3 model.
    (BatchSize) list of ragged (Tokens, 1024) float vectors
    """
    return await process_embedding_request(
        texts=body.texts,
        type=EmbeddingType.colbert,
    )


@app.post(
    "/bge-m3/all",
    response_model=Dict[str, Any],
    tags=["Embeddings"],
)
async def get_all_embeddings(
    body: EmbedRequest,
    request: Request,
):
    """
    Get all three embedding types from the mock BGE-M3 model.
    """
    return await process_embedding_request(
        texts=body.texts,
        type=EmbeddingType.all,
    )


@app.get("/")
def read_root():
    return {
        "message": "Mock BGE-M3 FastAPI wrapper is running.",
        "endpoints": [
            "POST /bge-m3/dense",
            "POST /bge-m3/lexical",
            "POST /bge-m3/colbert",
            "POST /bge-m3/all",
        ],
        "docs": "See /docs for API details.",
    }


if __name__ == "__main__":
    print(
        "Starting mock FastAPI server on http://0.0.0.0:8000"
    )
    uvicorn.run(app, host="0.0.0.0", port=8000)

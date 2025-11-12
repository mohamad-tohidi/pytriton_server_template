import numpy as np
from pytriton.decorators import batch
from pytriton.model_config import (
    ModelConfig,
    Tensor,
    DynamicBatcher,
)
from pytriton.triton import Triton, TritonConfig
from FlagEmbedding import BGEM3FlagModel
from typing import Dict, Literal, Union, List
import orjson as json

MODEL = BGEM3FlagModel(
    "BAAI/bge-m3", use_fp16=False, devices="cpu"
)
MAX_LENGTH = 8000


@batch
def infer_fn(texts: np.ndarray):
    py_txts = [t.decode("utf-8") for t in texts]
    out = MODEL.encode(
        py_txts,
        batch_size=len(py_txts),
        max_length=MAX_LENGTH,
        return_colbert_vecs=True,
        return_dense=True,
        return_sparse=True,
    )
    out: Dict[
        Literal[
            "dense_vecs", "lexical_weights", "colbert_vecs"
        ],
        Union[
            np.ndarray,
            List[Dict[str, float]],
            List[np.ndarray],
        ],
    ]

    dense_vecs_batch = out["dense_vecs"]

    lexical_json_list = [
        json.dumps(d).encode("utf-8")
        for d in out["lexical_weights"]
    ]
    lexical_weights_batch = np.array(
        lexical_json_list, dtype=np.bytes_
    )
    colbert_json_list = [
        json.dumps(arr.tolist()).encode("utf-8")
        for arr in out["colbert_vecs"]
    ]
    colbert_vecs_batch = np.array(
        colbert_json_list, dtype=np.bytes_
    )

    return {
        "dense_vecs": dense_vecs_batch,
        "lexical_weights": lexical_weights_batch,
        "colbert_vecs": colbert_vecs_batch,
    }


batching_config = ModelConfig(
    batching=True,
    max_batch_size=8,
    batcher=DynamicBatcher(
        max_queue_delay_microseconds=10_000,
        preferred_batch_size=[1, 2, 4, 8],
    ),
)


config = TritonConfig(
    http_port=8011,
    grpc_port=8012,
    metrics_port=8013,
    strict_readiness=True,
)

if __name__ == "__main__":
    print(
        "Starting PyTriton server for model 'bge-m3' on GPU (HTTP:8011, gRPC:8012)"
    )
    with Triton(config=config) as triton:
        triton.bind(
            model_name="bge-m3",
            infer_func=infer_fn,
            inputs=[
                Tensor(
                    name="texts",
                    dtype=np.bytes_,
                    shape=(1,),
                )
            ],
            outputs=[
                Tensor(
                    name="dense_vecs",
                    dtype=np.float32,
                    shape=(1024,),
                ),
                Tensor(
                    name="lexical_weights",
                    dtype=np.bytes_,
                    shape=(1,),
                ),
                Tensor(
                    name="colbert_vecs",
                    dtype=np.bytes_,
                    shape=(1,),
                ),
            ],
            config=batching_config,
        )
        triton.serve()

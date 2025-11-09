import numpy as np
from pytriton.decorators import batch
from pytriton.model_config import ModelConfig, Tensor, DynamicBatcher
from pytriton.triton import Triton, TritonConfig
from FlagEmbedding import BGEM3FlagModel

MODEL = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False, devices="cuda:0")


@batch
def infer_fn(texts: np.ndarray):
    py_texts = [text.decode("utf-8") for text in texts.flatten()]
    out = MODEL.encode(py_texts, batch_size=len(py_texts), max_length=512)
    dense = np.array(out["dense_vecs"], dtype=np.float32)
    return [dense]


batching_config = ModelConfig(
    batching=True,
    max_batch_size=8,
    batcher=DynamicBatcher(
        max_queue_delay_microseconds=10_000, preferred_batch_size=[1, 2, 4, 8]
    ),
)


config = TritonConfig(
    http_port=8011, grpc_port=8012, metrics_port=8013, strict_readiness=True
)

if __name__ == "__main__":
    print("Starting PyTriton server for model 'bge-m3' on GPU (HTTP:8011, gRPC:8012)")
    with Triton(config=config) as triton:
        triton.bind(
            model_name="bge-m3",
            infer_func=infer_fn,
            inputs=[Tensor(name="texts", dtype=np.bytes_, shape=(1,))],
            outputs=[Tensor(name="embeddings", dtype=np.float32, shape=(1024,))],
            config=batching_config,
        )
        # for more models
        # triton.bind(
        #     model_name="e5",
        #     ...
        # )
        triton.serve()

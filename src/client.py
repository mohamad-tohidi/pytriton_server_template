# the sample client code


import asyncio
import numpy as np
import orjson as json
from typing import List, Dict, Union
from pytriton.client import AsyncModelClient

SERVER_URL = "http://localhost:8011"
MODEL_NAME = "bge-m3"


async def infer_texts(
    texts: List[str],
) -> Dict[
    str,
    Union[
        np.ndarray, List[Dict[str, float]], List[np.ndarray]
    ],
]:
    """
    Asynchronous inference function for the BGE-M3 model server.

    Args:
        texts: List of input texts to encode.

    Returns:
        Dict with keys 'dense_vecs' (np.ndarray), 'lexical_weights' (List[Dict[str, float]]),
        and 'colbert_vecs' (List[np.ndarray]), mimicking the model's encode output.
    """
    async with AsyncModelClient(
        SERVER_URL, MODEL_NAME
    ) as client:
        # Step 1: Pre-process the texts
        # Convert list of strings to numpy array of bytes
        texts_array = np.array(
            [t.encode("utf-8") for t in texts],
            dtype=np.bytes_,
        ).reshape(-1, 1)

        # Step 2: Send the request to the server
        # Inputs dict with the pre-processed array
        inputs = {"texts": texts_array}
        result = await client.infer_batch(**inputs)

        # Step 3: Post-process the results
        # Dense vectors are already in usable form: (batch_size, 1024)
        dense_vecs = result["dense_vecs"]

        # Lexical weights: (batch_size, 1) bytes_ -> squeeze to (batch_size,) -> decode and load JSON
        lexical_bytes = result["lexical_weights"].squeeze(
            axis=1
        )
        lexical_weights = [
            json.loads(b) for b in lexical_bytes
        ]  # List[Dict[str, float]]

        # ColBERT vectors: (batch_size, 1) bytes_ -> squeeze to (batch_size,) -> decode, load JSON, convert to np.ndarray
        colbert_bytes = result["colbert_vecs"].squeeze(
            axis=1
        )
        colbert_vecs = [
            np.array(json.loads(b)) for b in colbert_bytes
        ]  # List[np.ndarray]

        # Return in the same format as MODEL.encode
        return {
            "dense_vecs": dense_vecs,
            "lexical_weights": lexical_weights,
            "colbert_vecs": colbert_vecs,
        }


# Example usage
async def main():
    texts = ["Hello, world!", "This is a test."]
    output = await infer_texts(texts)
    print(
        "Dense vectors shape:", output["dense_vecs"].shape
    )
    print(
        "First lexical weights:",
        output["lexical_weights"][0],
    )
    print(
        "First ColBERT vectors shape:",
        output["colbert_vecs"][0].shape,
    )


if __name__ == "__main__":
    asyncio.run(main())

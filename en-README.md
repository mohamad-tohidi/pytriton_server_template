# English Version README.md

# PyTriton Template

A minimal template to serve an embedding model using NVIDIA PyTriton (with dynamic batching), wrapped in FastAPI. Uses uv for package management, Docker for deployment, and Gitea Actions for CI/CD.

Supports any Hugging Face sentence-transformers model—swap via env vars.

Any other model can be served too. For example, we chose embedding models.

**Note**: If you have a GPU server and need CUDA in your model, install the `nvidia-container-toolkit`. You can run this [script](https://github.com/mohamad-tohidi/ai_server_setup/blob/main/install_nvidia_container_tool.sh) to automate it.

## Why Triton Server?

Triton Inference Server is a powerful tool for serving ML models in production.

Three key reasons why it's necessary:

1. **Cannot serve models directly on FastAPI in production**: FastAPI is great for APIs, but it struggles with heavy ML inference loads, GPU management, and scaling. Triton handles this efficiently.

2. **Dynamic Batching**: Automatically groups requests into batches for faster GPU processing, reducing latency and improving throughput.

3. **Stable and Solid for Production**: Built by NVIDIA, it's reliable, supports multiple frameworks (like PyTorch, TensorFlow), and has features like model ensembles and metrics.

## Components and Levels

This template has two main layers: Triton for core serving, and FastAPI as a wrapper.

- **Triton Served APIs**: Triton runs in the background and provides:
  - **HTTP**: For simple REST requests (default in this template).
  - **gRPC**: For high-performance, binary communication (faster for large data).
  - **Metrics**: Endpoint to monitor performance (e.g., latency, throughput) at `/metrics`.

- **FastAPI API**: A user-friendly wrapper around Triton's gRPC/HTTP. Clients can send simple JSON requests without installing PyTriton or Triton clients—just use curl or any HTTP tool. It makes integration easy for web apps.

## How to Serve Your Own Model

To adapt for your model (e.g., classification, generation):

1. **Modify `model.py`**: Load your model and update the `infer_fn` function. Change inputs/outputs to match (e.g., add more tensors). Keep `@batch` for dynamic batching.

2. **Update `api.py`**: Adjust Triton binding (inputs/outputs shapes, dtypes). Update FastAPI endpoints (e.g., request/response models) to fit your inference.

3. **Env Vars if Needed**: Add custom envs (e.g., for model paths) in Dockerfile and use in code.

That's it—test locally, then deploy!

## Quick Start

1. Install [uv](https://docs.astral.sh/uv/) if you don't have it.

   **Pro Tip**: If you're not using uv, stop this guide right now and learn it. It's not complicated, and it's great for long-term use since it's written in Rust and super fast!

2. Clone the repo and `cd` into it.

3. Run `uv sync` to install dependencies.

## Environment Variables

Customize without rebuilding by setting env vars at runtime (e.g., via `-e` in Docker or export locally). Defaults work for public models.

- `MODEL_NAME`: Hugging Face model (default: `all-MiniLM-L6-v2`).
- `HF_TOKEN`: For private/gated models (default: empty).
- `MAX_BATCH_SIZE`: Triton batch limit (default: `64`).
- `FASTAPI_PORT`: API port (default: `8080`).
- `UVICORN_WORKERS`: Uvicorn workers (default: `1`; keep low for GPUs).

Example local: `export MODEL_NAME="BAAI/bge-large-en-v1.5"`

**Note**: If you need to set env vars for your CI/CD process, go to Settings/Actions/Secrets in your Gitea site and configure them there.

## Server Setup

<details>
<summary>If you haven't set up CI/CD on your server, read this (click to expand)</summary>

The process is simple. First, get a token from Gitea/GitHub. This token proves you're the owner and allows your code to run here. Then, run a Docker container on your server. This container waits for your Actions!

Step 1: Get a Registration Token

The runner needs a token to connect securely to your Gitea instance.

Go to your Gitea repository and click Settings > Actions.

Find the Runners section.

Click Create new Runner. This will generate a registration token for you. Copy this token—you'll need it in the next step.

Note: You can also create runners at the Organization or Instance (Admin) level, which allows them to be shared across multiple repositories.

Step 2: Run the act_runner in Docker

On your server, run the following docker run command. This command downloads the act_runner image, starts it, and registers it with your Gitea instance all at once.

It mounts the Docker socket (/var/run/docker.sock) so your CI/CD jobs can build and run Docker containers.

It creates a volume (gitea-runner-data) to store its configuration.

Replace https://your-gitea.com and `YOUR_TOKEN_HERE` with your values.

```bash
docker run -d --restart=always \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v gitea-runner-data:/data \
  -e GITEA_INSTANCE_URL=https://your-gitea.com \
  -e GITEA_RUNNER_REGISTRATION_TOKEN=YOUR_TOKEN_HERE \
  -e GITEA_RUNNER_NAME=my-docker-runner \
  --name gitea_runner \
  gitea/act_runner:latest
```

After a few seconds, if you refresh the Settings > Actions > Runners page in Gitea, you should see your new runner with a green "Idle" status.

</details>

<details>
<summary>Local Development (click to expand)</summary>

1. Run: `uv run uvicorn pytriton_template.api:app --host 0.0.0.0 --port 8080 --reload`

Access at http://localhost:8080/docs for Swagger UI.

</details>

<details>
<summary>Docker Deployment (click to expand)</summary>

1. Build: `docker build -t pytriton_template .`
2. Run: `docker run --gpus all -p 8080:8080 pytriton_template`

For custom env: Add `-e MODEL_NAME="your-model" -e HF_TOKEN="your-token"`.

</details>

## Usage

POST to `/embed`:

```bash
curl -X POST http://localhost:8080/embed -H "Content-Type: application/json" -d '{"texts": ["hello world"]}'
```

Returns: `{"embeddings": [[0.1, 0.2, ...]]}`

Check health: `curl http://localhost:8080/health`

## CI/CD

Gitea Actions builds/tests Docker on push. Add secrets (e.g., HF_TOKEN) in repo settings for private models.

## Customize

- Edit `model.py` for different inference logic.
- Add endpoints in `api.py`.
- Fork and extend!


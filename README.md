
# قالب PyTriton


A minimal template to serve an embedding model using NVIDIA PyTriton (with dynamic batching), wrapped in FastAPI. Uses uv for package management, Docker for deployment, and Gitea Actions for CI/CD.

هر مدل دیگه‌ای هم می‌تونه سرو بشه. مثلاً ما مدل‌های امبدینگ رو انتخاب کردیم.

**نکته**:
اگر میخواهید که از GPU سرورتون استفاده کنید
باید ابزار 

`nvidia-container-toolkit`
در سرورتون نصب باشه

[run this in the server to install automatically](https://github.com/mohamad-tohidi/ai_server_setup/blob/main/install_nvidia_container_tool.sh) 

## چرا سرور Triton؟

سرور اینفرنس Triton یک ابزار قدرتمند برای سرو مدل‌های ML/DL در پروداکشنه.

اگر با این سرویس آشنا نیستید این [صفحه](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html) رو یک نگاه بیندازید.

**Scale**:

اگر منابع محاسباتی سرور دوبرابر بشود میتوانید دوبرابر حجم درخواست را پردازش کنید؟
آیا از نهایت منابع پردازشی سرور برای نهایت سرعت ممکن استفاده میکنید؟


**Dynamic Batching**:

کاربرهای مختلف درخواست هاشون رو به سرور شما میفرستند
شما تا یک بازه کوتاه (مثلا 100 میکرو ثانیه) صبر میکنی
تمام درخواست هایی که در این بازه برای شما فرستاده شده بودند رو یک Batch میکنی
و اون Batch رو به مدل میدی برای پردازش

نتیجه:
 سرعت بسیار بیشتر نسبت به پردازش های تک به تک.

...

این دوتا از دلایلی هست که من برای Production بیشتر سعی میکنم از TRT استفاده کنم.


## کامپوننت‌ها و سطوح
دو لایه داریم
در مرکز اصلی که ترایتون سرو شده
بعد یک لایه FastAPI حول اون میکشیم
که استفاده از مدل رو راحت تر تر بکنه
این تکنیک در داکیومنت های رسمی pytriton پیشنهاد شده است

به این [بخش](https://triton-inference-server.github.io/pytriton/latest/clients/) مراجعه کنید


**Triton**:

سه نقطه خروجی داره:

HTTP/gRPC/metrics


‍‍**fastapi**:

یک نقطه خروجی داره
که درواقع یک wrapper هست حول پورت gRPC ترایتون

دلیل اضافه کردن این API صرفا راحتی استفاده است





## چطور مدل خودتون رو سرو کنید

برای تطبیق با مدل خودتون (مثل کلاسفیکیشن، جنریشن):

**تغییر `model.py`**:

 مدلتون رو لود کنید و تابع `infer_fn` رو آپدیت کنید. ورودی/خروجی‌ها  بدید (مثل اضافه کردن تنسورهای بیشتر). `@batch` رو نگه دارید برای Batch داینامیک.

**آپدیت `api.py`**:

 بایندینگ Triton رو تنظیم کنید (شکل و dtype ورودی/خروجی). اندپوینت‌های FastAPI رو تغییر بدید (مثل مدل‌های درخواست/پاسخ) تا با اینفرنس‌تون جور بشه.

**env varها اگر لازم**:

 env varهای کاستوم اضافه کنید (مثل مسیر مدل) در Dockerfile و در کد استفاده کنید.

اول لوکال تست کنید
اگر همه چی اوکیه
بفرستید روی سرور!

## شروع سریع

1. [uv](https://docs.astral.sh/uv/) رو نصب کنید اگر ندارید.

   **نکته حرفه‌ای**: اگر از uv استفاده نمی‌کنید، همین الان این راهنما رو متوقف کنید و برید یادش بگیرید. اصلاً پیچیده نیست، و برای طول عمر مناسبه چون با Rust نوشته شده و فوق‌العاده سریعه!

2. ریپو رو کلون کنید و `cd` داخلش برید.

3. `uv sync` رو ران کنید تا وابستگی‌ها نصب بشن.

## متغیرهای محیطی

بدون rebuild سفارشی‌سازی کنید با تنظیم env varها در زمان ران (مثل `-e` در Docker یا export لوکال). دیفالت‌ها برای مدل‌های عمومی کار می‌کنن.

- `MODEL_NAME`: مدل Hugging Face (دیفالت: `all-MiniLM-L6-v2`).
- `HF_TOKEN`: برای مدل‌های خصوصی/قفل‌شده (دیفالت: خالی).
- `MAX_BATCH_SIZE`: حد باتچ Triton (دیفالت: `64`).
- `FASTAPI_PORT`: پورت API (دیفالت: `8080`).
- `UVICORN_WORKERS`: ورکرهای Uvicorn (دیفالت: `1`؛ برای GPUها کم نگه دارید).

مثال لوکال: `export MODEL_NAME="BAAI/bge-large-en-v1.5"`

**نکته**: اگر نیاز دارید env var تعریف کنید تا در CI/CD اعمال بشه، به بخش Settings/Actions/Secrets در سایت Gitea برید و تنظیمش کنید.

## تنظیم سرور

<details>
<summary>اگر سرورتون رو CI/CD نکردید این بخش رو بخونید (کلیک برای باز کردن)</summary>

فرآیند ساده است. ابتدا یک توکن از Gitea/GitHub میگیرید. این توکن نشون‌دهنده اینه که شما مالک سرورید و کدهای شما اجازه اجرا دارن. بعد، یک کانتینر Docker روی سرورتون ران می‌کنید. این کانتینر منتظر Actionهای شماست!

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
docker run -d --restart=always   -v /var/run/docker.sock:/var/run/docker.sock   -v ./gitea-runner-data:/data   -e GITEA_INSTANCE_URL=https://git.t.etratnet.ir   -e GITEA_RUNNER_REGISTRATION_TOKEN=*************   -e GITEA_RUNNER_NAME=my-docker-runner -e GITEA_RUNNER_LABELS="self-hosted,H100" --name gitea_runner   gitea/act_runner:latest                                                                      
```

After a few seconds, if you refresh the Settings > Actions > Runners page in Gitea, you should see your new runner with a green "Idle" status.

</details>

<details>
<summary>توسعه لوکال (کلیک برای باز کردن)</summary>

1. ران کنید: `uv run uvicorn pytriton_template.api:app --host 0.0.0.0 --port 8080 --reload`

به http://localhost:8080/docs برای Swagger UI برید.

</details>

<details>
<summary>دیپلوی Docker (کلیک برای باز کردن)</summary>

1. بیلد کنید: `docker build -t pytriton_template .`
2. ران کنید: `docker run --gpus all -p 8080:8080 pytriton_template`

برای env کاستوم: `-e MODEL_NAME="your-model" -e HF_TOKEN="your-token"` اضافه کنید.

</details>

## استفاده

POST به `/embed`:

```bash
curl -X POST http://localhost:8080/embed -H "Content-Type: application/json" -d '{"texts": ["hello world"]}'
```

برمی‌گردونه: `{"embeddings": [[0.1, 0.2, ...]]}`

چک هلث: `curl http://localhost:8080/health`

## CI/CD

Gitea Actions روی پوش بیلد/تست Docker می‌کنه. secretها (مثل HF_TOKEN) رو در تنظیمات ریپو برای مدل‌های خصوصی اضافه کنید.

## سفارشی‌سازی

- `model.py` رو برای منطق اینفرنس متفاوت ویرایش کنید.
- اندپوینت‌ها رو در `api.py` اضافه کنید.
- فورک کنید و گسترش بدید!
FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface \
    HF_ENDPOINT=https://hf-mirror.com

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-deps \
    torch==2.3.1+cu121 torchvision==0.18.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121
RUN pip install -r requirements.txt

COPY *.py ./

CMD ["python", "02_using_your_model.py"]

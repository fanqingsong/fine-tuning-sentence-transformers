FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface \
    HF_ENDPOINT=https://hf-mirror.com

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-deps \
    torch==2.3.1+cpu torchvision==0.18.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu
RUN pip install -r requirements.txt

COPY 01_tuning_your_model.py 02_using_your_model.py ./

CMD ["python", "02_using_your_model.py"]

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip ffmpeg git \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace/MiniCPM-V
COPY web_demos/minicpm-o4.5/requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements.txt
COPY . /workspace/MiniCPM-V
EXPOSE 32560
CMD ["python3", "-m", "uvicorn", "--app-dir", "web_demos/minicpm-o4.5", "app.server:app", "--host", "0.0.0.0", "--port", "32560"]


FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONPATH=/workspace
ENV MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /workspace

RUN pip install --upgrade pip setuptools wheel

COPY requirements.txt .
# torch and torchvision come from the base image (CUDA 12.4 build).
# requirements.txt must NOT re-list them or pip may swap in a CPU-only wheel.
RUN pip install -r requirements.txt

COPY . .

CMD ["bash"]

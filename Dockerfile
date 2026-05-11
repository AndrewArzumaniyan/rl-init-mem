FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /workspace

RUN pip install --upgrade pip setuptools wheel

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126
ARG TORCH_VERSION=2.10.0
RUN pip install --index-url "${TORCH_INDEX_URL}" torch=="${TORCH_VERSION}"

ARG AIHWKIT_GPU_WHEEL_URL=https://aihwkit-gpu-demo.s3.us-east.cloud-object-storage.appdomain.cloud/aihwkit-1.1.0-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
RUN pip install --extra-index-url "${TORCH_INDEX_URL}" "${AIHWKIT_GPU_WHEEL_URL}"

COPY requirements.txt .
RUN pip install -r requirements.txt

RUN python -c "import aihwkit, torch; print(aihwkit.__version__, torch.__version__)"

COPY . .

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

FROM nvcr.io/nvidia/tensorrt:20.03-py3

LABEL maintainer="Jin Yeom"
LABEL version="0.0.1"

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update -y && apt-get install -y build-essential libglib2.0-0 \
    libsm6 libxext6 libxrender-dev libpng-dev libjpeg-dev libopencv-dev \
    python3-opencv ca-certificates pkg-config git curl wget git cmake \
    ffmpeg vim && \
    rm -rf /var/lib/apt/lists/*

# install packages
RUN pip3 install black onnx onnxruntime onnx-simplifier jupyter

# install Detectron2
ENV FORCE_CUDA "1"
ENV TORCH_CUDA_ARCH_LIST "Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
RUN pip3 install \
    torch \
    torchvision \
    cython \
    'git+https://github.com/facebookresearch/fvcore.git' \
    'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
RUN git clone https://github.com/facebookresearch/detectron2.git /detectron2 && \
    pip3 install -e /detectron2

WORKDIR /

EXPOSE 8888 6006
CMD jupyter notebook --no-browser --ip '0.0.0.0' --allow-root
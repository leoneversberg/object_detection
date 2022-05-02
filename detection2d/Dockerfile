FROM ubuntu:focal
# Packages
#
# Ubuntu 20.04: https://github.com/tianon/docker-brew-ubuntu-core/blob/74249faf47098bef2cedad89696bfd1ed521e019/focal/Dockerfile
# MMDetection: https://mmdetection.readthedocs.io/en/latest/get_started.html#installation
# PyTorch: https://pytorch.org/get-started/locally/
ARG PYTORCH="1.7.0"
ARG CUDA="11.0"
ARG CUDNN="8"
ENV NVIDIA_REQUIRE_CUDA "cuda>=11.0"
ENV NVIDIA_DRIVER_CAPABILITIES compute
ENV FORCE_CUDA="1"
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

# setup timezone
RUN echo 'Etc/UTC' > /etc/timezone && \
    ln -s /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
    apt-get update && \
    apt-get install -q -y --no-install-recommends tzdata && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y wget ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    python3-pip \
    #libopencv-dev python3-opencv \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install torch==1.7.0+cu110 torchvision==0.8.1+cu110 torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html \
    && pip3 install mmcv-full==1.2.5 -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html # Install MMCV

# Install MMDetection
#RUN conda clean --all
RUN git clone --depth 1 --branch v2.11.0 https://github.com/open-mmlab/mmdetection.git /mmdetection
WORKDIR /mmdetection
RUN pip3 install -r requirements/build.txt
RUN pip3 install --no-cache-dir -e .

# get pre-trained ResNet50 and Faster R-CNN
WORKDIR /mmdetection/checkpoints
RUN wget https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
RUN wget https://download.pytorch.org/models/resnet50-19c8e357.pth 

RUN pip install -U albumentations

## copy our mmdetection configs
COPY . /mmdetection/
WORKDIR /mmdetection
#RUN chmod +x train.sh
CMD ["/mmdetection/train.sh"]


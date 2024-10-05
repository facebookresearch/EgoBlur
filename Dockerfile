FROM homebrew/ubuntu22.04

USER root

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.10 python3.10-dev python3-pip && \
    apt-get clean

RUN ln -fs /usr/share/zoneinfo/UTC /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata && \
    apt-get install -y tzdata

RUN rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

COPY /requirements.txt . /workspace/

WORKDIR /workspace

RUN pip install --upgrade pip

RUN pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116

RUN pip install -r /workspace/requirements.txt

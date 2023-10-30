FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
ENV DEBIAN_FRONTEND=noninteractive
ENV CMAKE_INSTALL_PREFIX=/usr/local

# Install apt dependencies
RUN apt-get update && \
    apt-get install -y \
        gpg \
        wget \
        git \
        vim \
        # libboost-all-dev \
        # libpcl-dev \
        freeglut3-dev \
        libglib2.0-dev \
        python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# install vscode 
RUN wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > packages.microsoft.gpg && \
    install -D -o root -g root -m 644 packages.microsoft.gpg /etc/apt/keyrings/packages.microsoft.gpg && \
    sh -c 'echo "deb [arch=amd64,arm64,armhf signed-by=/etc/apt/keyrings/packages.microsoft.gpg] https://packages.microsoft.com/repos/code stable main" > /etc/apt/sources.list.d/vscode.list' && \
    rm -f packages.microsoft.gpg && \
    apt -y install apt-transport-https && \
    apt update && \
    apt -y install code

# install cuda
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin && \
    mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600 && \
    wget https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda-repo-ubuntu2004-11-7-local_11.7.0-515.43.04-1_amd64.deb && \
    dpkg -i cuda-repo-ubuntu2004-11-7-local_11.7.0-515.43.04-1_amd64.deb && \
    cp /var/cuda-repo-ubuntu2004-11-7-local/cuda-*-keyring.gpg /usr/share/keyrings/ && \
    apt-get update && \
    apt-get -y install cuda



ENV CMAKE_INSTALL_PREFIX=/usr/local
ENV DEPS_DIR=/root/deps

WORKDIR ${DEPS_DIR}

ARG NUM_THREADS=6
ARG READ_TOKEN

# Clone, build, and install OpenCV @ 4.2.0
# RUN git clone https://github.com/opencv/opencv.git && \
#     cd opencv && \
#     git checkout 4.2.0 && \
#     mkdir build && \
#     cd build && \
#     cmake \
#         -D CMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX} \
#         -D CMAKE_BUILD_TYPE=Release \
#         .. && \
#     make -j${NUM_THREADS} install && \
#     make install && \
#     cd ${DEPS_DIR} && \
#     rm -rf opencv

ENV CMAKE_INSTALL_PREFIX=/usr/local


ENV CUDA_HOME=/usr/local/cuda-11.7
ENV CUDA_PATH=/usr/local/cuda-11.7
ENV TCNN_CUDA_ARCHITECTURES=86
# ENV CUDACXX=/usr/local/cuda/bin/nvcc

# Set the working directory
WORKDIR /root/repos


# Clone and install the packages and tcnn
RUN git clone https://${READ_TOKEN}:x-oauth-basic@github.com/xyp8023/neusis_ngp.git && \
    cd neusis_ngp && \
    pip install -r requirements.txt && \
    pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch




# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

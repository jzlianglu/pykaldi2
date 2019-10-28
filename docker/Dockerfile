FROM nvidia/cuda:9.0-devel-ubuntu16.04

ENV PYTORCH_VERSION=1.2.0
ENV CUDNN_VERSION=7.4.1.5-1+cuda9.0
ENV NCCL_VERSION=2.3.7-1+cuda9.0

# Python 2.7 or 3.5 is supported by Ubuntu Xenial out of the box
ARG python=3.5
ENV PYTHON_VERSION=${python}

RUN apt-get update && apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
        build-essential \
        cmake \
        git \
        curl \
        vim \
        wget \
        autoconf \
        automake \
        unzip \
        pkg-config \
        g++ \
        graphviz \
        libatlas3-base \
        libtool \
        subversion \
        make \
        zlib1g-dev \
        sox \
        python2.7 \
        python3 \
        ca-certificates \
        libcudnn7=${CUDNN_VERSION} \
        libnccl2=${NCCL_VERSION} \
        libnccl-dev=${NCCL_VERSION} \
        libjpeg-dev \
        libpng-dev \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev

RUN ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

# Dockerfile for building PyKaldi image from Ubuntu 16.04 image      
# Install necessary Python packages                                  
RUN pip install --upgrade pip \                                      
    numpy \                                                          
    setuptools \                                                     
    pyparsing \                                                      
    jupyter \
    editdistance \                                                        
    ninja                                                            
                                                                     
RUN mkdir /tmp/pykaldi && \                                          
    git clone https://github.com/pykaldi/pykaldi.git && \            
    cd pykaldi/tools && \                                            
    ./install_protobuf.sh && \                                       
    ./install_clif.sh && \                                           
    ./install_kaldi.sh && \                                          
    cd .. && \                                                       
    python setup.py install                                     


# Install PyTorch
RUN pip install h5py torch==${PYTORCH_VERSION} torchvision

# Install Open MPI
RUN mkdir /tmp/openmpi && \
    cd /tmp/openmpi && \
    wget https://www.open-mpi.org/software/ompi/v4.0/downloads/openmpi-4.0.0.tar.gz && \
    tar zxf openmpi-4.0.0.tar.gz && \
    cd openmpi-4.0.0 && \
    ./configure --enable-orterun-prefix-by-default && \
    make -j $(nproc) all && \
    make install && \
    ldconfig && \
    rm -rf /tmp/openmpi

# Install Horovod, temporarily using CUDA stubs
RUN ldconfig /usr/local/cuda-9.0/targets/x86_64-linux/lib/stubs && \
    HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_WITH_PYTORCH=1 pip install --no-cache-dir horovod && \
    ldconfig

# Install OpenSSH for MPI to communicate between containers
RUN apt-get install -y --no-install-recommends openssh-client openssh-server && \
    mkdir -p /var/run/sshd

# Allow OpenSSH to talk to containers without asking for confirmation
RUN cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new && \
    echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new && \
    mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config

# Download examples
RUN apt-get install -y --no-install-recommends subversion && \
    svn checkout https://github.com/horovod/horovod/trunk/examples && \
    rm -rf /examples/.svn

RUN pip install pyyaml scipy matplotlib 

RUN pip install soundfile
WORKDIR "/examples"

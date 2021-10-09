FROM centos:7

# Update centos and install basic dependencies
RUN yum update -y \
  && yum install \
    centos-release-scl \
    gcc-c++ \
    java-1.8.0-openjdk-devel \
    git \
    wget \
    devtoolset-7 \
    bzip2 \
    maven \
    -y \
   && scl enable devtoolset-7 bash

# Install cmake3
RUN wget https://github.com/Kitware/CMake/releases/download/v3.21.3/cmake-3.21.3-linux-x86_64.sh \
  && bash cmake-3.21.3-linux-x86_64.sh --prefix=/usr/local --exclude-subdir --skip-license

# Install anaconda3
RUN wget https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh \
  && bash Anaconda3-2019.03-Linux-x86_64.sh -b && \
    echo "export PATH="/root/anaconda3/bin:$PATH"" >> ~/.bashrc && \
    /bin/bash -c "source ~/.bashrc"
ENV PATH /root/anaconda3/bin:$PATH

# Install Python libraries
RUN conda install pybind11 tensorflow \
  && pip install -U pip \
  && pip install ray[rllib]==1.3.0

# Create working directory
RUN mkdir -p app
WORKDIR app

# Set entry point
CMD ["mvn", "clean", "install", "-Djavacpp.platform=linux-x86_64"]

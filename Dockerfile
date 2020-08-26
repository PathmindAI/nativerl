FROM centos:7

# Update centos and install basic dependencies
RUN yum update -y
RUN yum install centos-release-scl -y
RUN yum install gcc-c++ java-1.8.0-openjdk-devel git wget devtoolset-7 bzip2 maven -y
RUN scl enable devtoolset-7 bash

# Install anaconda3
RUN wget https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh
RUN bash Anaconda3-2019.03-Linux-x86_64.sh -b && \
    echo "export PATH="/root/anaconda3/bin:$PATH"" >> ~/.bashrc && \
    /bin/bash -c "source ~/.bashrc"
ENV PATH /root/anaconda3/bin:$PATH

# Install Python libraries
RUN conda install pybind11 tensorflow
RUN pip install ray[rllib]

# Create working directory
RUN mkdir -p app
WORKDIR app

# Set entry point
CMD ["mvn", "clean", "install"]

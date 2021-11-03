FROM ubuntu:20.04

# Update ubuntu
RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive \
    apt-get install -y \
    wget

# Install anaconda3
RUN wget https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh \
    && bash Anaconda3-2019.03-Linux-x86_64.sh -b && \
    echo "export PATH="/root/anaconda3/bin:$PATH"" >> ~/.bashrc && \
    /bin/bash -c "source ~/.bashrc"
ENV PATH /root/anaconda3/bin:$PATH

ADD requirements.txt .
RUN conda install -y pybind11 conda-pack  && \
    pip install -U pip && \
    pip install -r requirements.txt

RUN mkdir -p app
WORKDIR app

# Black magic. Conda pack fails the first time, so we have to run list first.
RUN conda list


# Set entry point
CMD /bin/bash -c "source ~/.bashrc" && \
    conda pack -f -o rllibpack.tar.gz

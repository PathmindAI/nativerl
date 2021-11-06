FROM ubuntu:20.04

# Update ubuntu
RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive \
    apt-get install -y \
    build-essential libcurl4-gnutls-dev libxml2-dev libssl-dev \
    wget

RUN useradd -ms /bin/bash rllibpack

USER rllibpack
WORKDIR /home/rllibpack

# Install anaconda3
RUN wget https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh \
    && bash Anaconda3-2019.03-Linux-x86_64.sh -b && \
    echo "export PATH="/home/rllibpack/anaconda3/bin:$PATH"" >> ~/.bashrc && \
    /bin/bash -c "source ~/.bashrc"
ENV PATH /home/rllibpack/anaconda3/bin:$PATH

RUN conda install -y pybind11
ADD requirements.txt .
RUN pip install -U pip && \
    pip install -r requirements.txt

RUN mkdir -p out
WORKDIR /home/rllibpack/

RUN conda install -c conda-forge conda-pack
# Black magic. Conda pack fails the first time, so we have to run list first.
RUN conda list

RUN conda pack -f -o rllibpack.tar.gz
RUN chmod 644 rllibpack.tar.gz

CMD ["cp", "/home/rllibpack/rllibpack.tar.gz", "/home/rllibpack/out/", "-v"]

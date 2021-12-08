#https://blog.ceshine.net/post/replicate-conda-environment-in-docker/
# https://towardsdatascience.com/making-docker-and-conda-play-well-together-eda0ff995e3c
# build image as "docker build --tag <name> . "
# run as docker run -it -v ~/Documents/Mapping/sample_docker_data/data:/home/docker/mesh_class_labeling/data <name>  #-it options give you interactive shell

FROM ubuntu:20.04

ARG CONDA_PYTHON_VERSION=3
ARG CONDA_DIR=/opt/conda
ARG USERNAME=docker
ARG USERID=1000

RUN apt-get update && \
    apt-get install -y --no-install-recommends git wget unzip bzip2 sudo build-essential ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

## Install miniconda
ENV PATH $CONDA_DIR/bin:$PATH
RUN wget --quiet https://repo.continuum.io/miniconda/Miniconda$CONDA_PYTHON_VERSION-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    echo 'export PATH=$CONDA_DIR/bin:$PATH' > /etc/profile.d/conda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm -rf /tmp/*

# Create the user
RUN useradd --create-home -s /bin/bash --no-user-group -u $USERID $USERNAME && \
    chown $USERNAME $CONDA_DIR -R && \
    adduser $USERNAME sudo && \
    echo "$USERNAME ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

USER $USERNAME
WORKDIR /home/$USERNAME

RUN conda install -y mamba -c conda-forge

#COPY ./environment.yml .
COPY . .
RUN sudo chown $USERNAME . -R
RUN mamba env update --file ./environment.yml &&\
    conda clean -tipy

# For interactive shell
RUN conda init bash
RUN echo "conda activate base" >> /home/$USERNAME/.bashrc
RUN bash

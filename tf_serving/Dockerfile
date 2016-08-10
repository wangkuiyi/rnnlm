FROM cxwangyi/bazel

MAINTAINER Yi Wang

RUN apt-get update && apt-get install -y \
        build-essential \
        curl \
        git \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        python-dev \
        python-numpy \
        python-pip \
        software-properties-common \
        swig \
        zip \
        zlib1g-dev


WORKDIR /
RUN git clone --recurse-submodules https://github.com/tensorflow/serving

WORKDIR /serving/tensorflow
RUN ./configure

WORKDIR /serving
bazel build tensorflow_serving/...


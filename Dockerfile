# Using official Python image as base image
FROM python:3.8 AS root

# Set environment variables
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Initialize working directory
RUN mkdir -p /project_fused && ls -la /project_fused
WORKDIR /project_fused

# Turn off interactivity
ENV DEBIAN_FRONTEND=noninteractive

# Install useful tools
RUN apt-get update && apt-get install -y \
    sudo \
    git \
    git-lfs \
    ssh \
    curl \
    nano \
    vim \
    less \
    usbutils \
    protobuf-compiler \
    autoconf \
    libtool \
    rsync \
    libboost-all-dev \
    openssh-client \
    libgl1-mesa-glx \
    x11-apps \
    v4l-utils \
    kmod

# Install python/pip
RUN apt-get update && apt-get install -y python3-pip

# Install other tools using pip
RUN pip install --upgrade pip
RUN pip install \
    scikit-learn \
    scipy \
    pandas \
    opencv-python \
    ipython \
    matplotlib \
    numpy \
    mediapipe \
    --timeout=1200

# Add docker user with same UID and GID as your host system
# (copied from https://code.visualstudio.com/remote/advancedcontainers/add-nonroot-user#_creating-a-nonroot-user)
FROM root AS image-nonroot
ARG USERNAME=docker
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME
# Switch from root to user
USER $USERNAME
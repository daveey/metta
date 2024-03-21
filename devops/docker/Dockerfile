FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-devel

WORKDIR /workspace

# Set the timezone
RUN ln -snf /usr/share/zoneinfo/America/Los_Angeles /etc/localtime && echo America/Los_Angeles > /etc/timezone

RUN apt-get update && apt-get install -y ninja-build git sudo wget vim screen
RUN apt-get install -y vulkan-tools libvulkan-dev vulkan-validationlayers-dev spirv-tools
RUN apt-get install -y xdg-utils python3-opencv
RUN apt upgrade -y libopenblas-dev
RUN apt-get install -y git-lfs
RUN git lfs install

# Install Vulkan SDK
RUN sudo apt-get install -y pkg-config libxcb1-dev libx11-dev
RUN sudo apt-get install -y wayland-protocols libwayland-dev libxrandr-dev
RUN wget https://sdk.lunarg.com/sdk/download/1.3.224.1/linux/vulkansdk-linux-x86_64-1.3.224.1.tar.gz
RUN tar -zxvf vulkansdk-linux-x86_64-1.3.224.1.tar.gz
WORKDIR /workspace/1.3.224.1
RUN yes | ./vulkansdk

# Configure screen
RUN echo "defscrollback 10000" >> /root/.screenrc && \
    echo "termcapinfo xterm* ti@:te@" >> /root/.screenrc

# Install conan and libstd to build griddly
RUN conda install -c conda-forge libstdcxx-ng
RUN pip install conan==1.59.0

# Checkout metta and submodules
RUN git clone --recursive https://github.com/daveey/metta.git metta

WORKDIR /workspace/metta
RUN ./sandbox/setup.sh


# ENV VULKAN_SDK=/workspace/1.3.224.1/x86_64

FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

# Install dependencies
RUN apt-get update && apt-get install -y ninja-build git

WORKDIR /workspace

# Install Griddly
RUN git clone https://github.com/Bam4d/Griddly.git griddly
RUN pip install conan==1.59.0

WORKDIR /workspace/griddly
RUN conan install deps/conanfile.txt --profile default --profile deps/build.profile -s build_type=Release --build missing -if build
RUN cmake . -B build -GNinja -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake
RUN cmake --build build --config Release
RUN cd python && pip install -e .

# Install sample-factory
RUN pip install sample-factory

# Install vulkan
RUN apt-get install -y vulkan-tools libvulkan-dev vulkan-validationlayers-dev spirv-tools
RUN apt-get install -y sudo wget vim

WORKDIR /workspace/
RUN wget https://sdk.lunarg.com/sdk/download/1.3.224.1/linux/vulkansdk-linux-x86_64-1.3.224.1.tar.gz
RUN tar -zxvf vulkansdk-linux-x86_64-1.3.224.1.tar.gz

# RUN cd 1.3.268.0/ && yes | ./vulkansdk

ENV VULKAN_SDK=/workspace/1.3.224.1/x86_64
RUN touch /root/.no_auto_tmux
RUN ln -snf /usr/share/zoneinfo/America/Los_Angeles /etc/localtime && echo America/Los_Angeles > /etc/timezone

RUN apt-get install -y xdg-utils python3-opencv
RUN apt upgrade -y libopenblas-dev
RUN pip uninstall -y numpy
RUN pip install numpy

RUN pip install aws
RUN git clone https://github.com/daveey/metta.git


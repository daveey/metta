FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel


# Install Griddly
RUN pip install conan==1.59.0

RUN
RUN
RUN cd python && pip install -e .

WORKDIR /workspace/
RUN wget https://sdk.lunarg.com/sdk/download/1.3.224.1/linux/vulkansdk-linux-x86_64-1.3.224.1.tar.gz
RUN tar -zxvf vulkansdk-linux-x86_64-1.3.224.1.tar.gz

# RUN cd 1.3.268.0/ && yes | ./vulkansdk

ENV VULKAN_SDK=/workspace/1.3.224.1/x86_64
RUN touch /root/.no_auto_tmux

RUN pip uninstall -y numpy
RUN pip install numpy

WORKDIR /workspace/metta
RUN pip install -r requirements.txt

WORKDIR /workspace/
# Install screen
RUN apt-get update && apt-get install -y screen

# Configure screen
RUN echo "defscrollback 10000" >> /root/.screenrc && \
    echo "termcapinfo xterm* ti@:te@" >> /root/.screenrc

# Automatically launch screen
RUN echo 'if [ -z "$STY" ]; then exec screen -R; fi' >> /root/.bashrc

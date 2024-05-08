FROM daveey/metta-base

RUN echo "Cache breaker: $(date)" > cache-breaker.txt

# Checkout metta and submodules
WORKDIR /workspace/
RUN date
RUN git clone --recursive https://github.com/daveey/metta.git metta
WORKDIR /workspace/metta

RUN conda create -n metta python=3.11.7
RUN echo "source activate metta" > ~/.bashrc

RUN /bin/bash -c "source activate metta && pip install --no-cache-dir -e ."

# Download the latest baseline model
RUN python -m sample_factory.huggingface.load_from_hub -r metta-ai/baseline -d /workspace/metta-baseline
RUN ln -s /workspace/metta-baseline/baseline/checkpoint_p0/$(ls /workspace/metta-baseline/baseline/checkpoint_p0 | tail -n1) /workspace/metta-baseline/latest


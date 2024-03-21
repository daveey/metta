FROM daveey/metta-base

# Checkout metta and submodules
WORKDIR /workspace/
RUN date
RUN git clone --recursive https://github.com/daveey/metta.git metta
WORKDIR /workspace/metta
RUN pip install -e .

# Download the latest baseline model
RUN python -m sample_factory.huggingface.load_from_hub -r metta-ai/baseline -d /workspace/metta-baseline
RUN ln -s /workspace/metta-baseline/baseline/checkpoint_p0/$(ls /workspace/metta-baseline/baseline/checkpoint_p0 | tail -n1) /workspace/metta-baseline/latest


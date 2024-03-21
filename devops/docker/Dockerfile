FROM daveey/meta-base:latest

# Checkout metta and submodules
WORKDIR /workspace/
RUN git clone --recursive https://github.com/daveey/metta.git metta
WORKDIR /workspace/metta
RUN pip install -e .


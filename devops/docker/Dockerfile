FROM daveey/metta-base

# Checkout metta and submodules
WORKDIR /workspace/
RUN git clone --recursive https://github.com/daveey/metta.git metta
WORKDIR /workspace/metta
RUN pip install -e .


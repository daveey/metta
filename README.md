# Installation

1. create conda environment

```
conda create -n meta python=3.11
conda activate meta
```

2. install requirements

```
pip install -r requirements.txt
```

3. install griddly (from source)

```
git clone https://github.com/Bam4d/Griddly.git griddly
pip install conan==1.59.0
conan install deps/conanfile.txt --profile default --profile deps/build.profile -s build_type=Release --build missing -if build
cmake . -B build -GNinja -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake
cmake --build build --config Release
cd python && pip install -e .
```

4. install vulkan sdk from https://vulkan.lunarg.com/sdk/home (1.3.224.1)

# Download trained models from s3

```
aws s3 sync s3://metta-ai/vast/workspace/metta/train_dir ./train_dir
```

# Run evaluation
```
python -m envs.griddly.enjoy \
    --device=cpu \
    --train_dir=./train_dir/ \
    --fps=10 --max_num_frames=1000 \
    --eval_env_frameskip=1 \
    --env=GDY-Forage \
    --experiment=g.pbt.4090
```

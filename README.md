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
## Griddly prerequisites:
1. Ninja (`brew install ninja` in MacOS)

```
git clone https://github.com/Bam4d/Griddly.git griddly

cd griddly

pip install conan==1.59.0

conan install deps/conanfile.txt --profile default --profile deps/build.profile -s build_type=Release --build missing -if build

cmake . -B build -GNinja -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake -DCMAKE_CXX_FLAGS=-w

cmake --build build --config Release

cd python && pip install -e .
```

4. install vulkan sdk from https://vulkan.lunarg.com/sdk/home (1.3.224.1)

5. install sample-factory (from source)

```
git clone https://github.com/daveey/sample-factory.git
cd sample-factory && pip install -e .
```

### Troubleshooting
#### conan installation
In case of an an "Invalid setting" error when running `pip install`, e.g., :
```
"ERROR: Invalid setting '15' is not a valid 'settings.compiler.version' value.
Possible values are ['5.0', ..., '14.0']
Read "http://docs.conan.io/en/latest/faq/troubleshooting.
html#error-invalid-setting"
```
Follow these steps (tested on MacOS):
1. open ~/.conan/settings.yml
2. Inside the file, go to `apple-clang` -> `version` and add missing value ("15" in the example above) is in the `version` list
3. For more information: https://docs.conan.io/en/1.16/faq/troubleshooting.html




# Run evaluation
```
python -m envs.griddly.enjoy \
    --device=cpu \
    --train_dir=./train_dir/ \
    --fps=10 --max_num_frames=1000 \
    --eval_env_frameskip=1 \
    --env=GDY-OrbWorld \
    --experiment=g.pbt.4090
```

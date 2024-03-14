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

    #### Griddly prerequisites:
    * Ninja (`brew install ninja` in MacOS)


    #### Installing Griddly:
    ```
    git clone https://github.com/daveey/Griddly.git griddly

    cd griddly

    pip install conan==1.59.0

    conan install deps/conanfile.txt --profile default --profile deps/build.profile -s build_type=Release --build missing -if build

    cmake . -B build -GNinja -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake -DCMAKE_CXX_FLAGS=-w

    cmake --build build --config Release

    cd python && pip install -e .
    cd ../..
    ```

4. install vulkan sdk from https://vulkan.lunarg.com/sdk/home (1.3.224.1)

5. install sample-factory (from source)

    ```
    git clone https://github.com/daveey/sample-factory.git
    cd sample-factory && pip install -e .
    cd ..
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
./evals/a20_40x40_rich.sh --experiment=p2.a100.batch.17.lr.e3 --train_dir=model_checkpoints
```

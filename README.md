# Installation

```
conda create -n metta python=3.11.7
conda activate metta
pip install -e .
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

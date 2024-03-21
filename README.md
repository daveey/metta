# Installation


### Install the Vulkan SDK

<details>
<summary>Mac</summary>

[Download Vulkan SDK for Mac](https://sdk.lunarg.com/sdk/download/1.3.224.1/mac/vulkansdk-macos-1.3.224.1.dmg)

</details>

<details>
<summary>Windows</summary>

[Download Vulkan SDK for Windows](https://sdk.lunarg.com/sdk/download/1.3.224.1/windows/VulkanSDK-1.3.224.1-Installer.exe)

</details>

<details>
<summary>Linux</summary>

```bash
mkdir ~/vulkan
cd ~/vulkan
wget https://sdk.lunarg.com/sdk/download/1.3.224.1/linux/vulkansdk-linux-x86_64-1.3.224.1.tar.gz
tar -zxvf vulkansdk-linux-x86_64-1.3.224.1.tar.gz
cd 1.3.224.1
yes | ./vulkansdk
echo 'source ~/1.3.224.1/setup-env.sh' >> ~/.bashrc
```
</details>

### Install the requirements
```
conda create -n metta python=3.11.7
conda activate metta
pip install -e .
```


# Evaluate a baseline model

### Setup HuggingFace Access

1. Create a HuggingFace account
2. Create a token in your account settings
3. Run `huggingface-cli login` and paste the token

### Download a baseline model

```
python -m sample_factory.huggingface.load_from_hub -r daveey/metta -d ./train_dir/
```

### Run the evaluation

```
./evals/a20_40x40_rich.sh --experiment=metta --no_render
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


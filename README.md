# Metta AI

Metta AI is an open-source research project that investigates the emergence of cooperation and alignment in multi-agent AI systems. By creating a model organism for complex multi-agent gridworld environments, the project aims to study the impact of social dynamics, such as kinship and mate selection, on the learning and cooperative behaviors of AI agents.

The project introduces a novel reward-sharing mechanism that mimics familial bonds and mate selection, allowing researchers to observe the evolution of complex social behaviors and cooperation among AI agents. The highly performant, open-ended game rules provide a rich environment for studying these behaviors and their potential implications for AI alignment.

Metta AI explores the hypothesis that emotional bonds and social dynamics, akin to love in biological systems, may play a crucial role in the development of cooperative AGI and AI alignment. By investigating this concept in a controlled multi-agent setting, the project seeks to contribute to the broader discussion on the path towards safe and beneficial AGI.

AI researchers will find Metta AI to be a valuable platform for studying the emergence of cooperation, social dynamics, and their potential impact on AI alignment. The project's modular design and open-source nature make it easy for researchers to adapt and extend the platform to investigate their own hypotheses in this domain.


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
Metta uses griddly and sample factory repos as submodules. To fetch the latest versions of these projects use the following commands:
```
git submodule init
git submodule update
```
Start installing the requiruments:
```
conda create -n metta python=3.11.7
conda activate metta
pip install -e .
```

### (Optional) Setup HuggingFace Access

1. Create a HuggingFace account
2. Create a token in your account settings
3. Run `huggingface-cli login` and paste the token

# Running a Simulation

### Download a baseline model
To download the model files, install git-lfs first, then run the following commands:

```
git lfs install
./devops/load_model.sh baseline
```

### Run the evaluation

```
./evals/a20_40x40_rich.sh --experiment=baseline --no_render
```

# Training a Model

### Run the training

```
 ./trainers/a5_25x25_simple.sh --experiment=my_experiment
```


### Run the training from a baseline model
```
./trainers/a5_25x25_simple.sh --experiment=my_experiment --init_checkpoint_path=./train_dir/baseline/$(ls train_dir/baseline/checkpoint_p0 | tail -n1)
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


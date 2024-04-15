import hydra
from omegaconf import OmegaConf

from rl_framework.sample_factory import SampleFactoryFramework

@hydra.main(version_base=None, config_path="../configs", config_name="configs")
def main(cfg):
    print(OmegaConf.to_yaml(cfg))
    framework = SampleFactoryFramework(cfg)
    framework.train()

if __name__ == "__main__":
    main()

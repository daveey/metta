from functools import partial
import hydra
from omegaconf import OmegaConf

from rl_framework.sample_factory.sample_factory import SampleFactoryFramework

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):
    print(OmegaConf.to_yaml(cfg))
    framework = hydra.utils.instantiate(cfg.framework, cfg, _recursive_=False)
    if cfg.cmd == "train":
        framework.train()
    if cfg.cmd == "evaluate":
        framework.evaluate()

if __name__ == "__main__":
    main()

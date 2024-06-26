from functools import partial
import os
import hydra
from omegaconf import OmegaConf
from rich import traceback

from rl_framework.sample_factory.sample_factory import SampleFactoryFramework

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):
    traceback.install(show_locals=False)
    print(OmegaConf.to_yaml(cfg))
    framework = hydra.utils.instantiate(cfg.framework, cfg, _recursive_=False)

    try:
        if cfg.cmd == "train":
            framework.train()
        if cfg.cmd == "evaluate":
            framework.evaluate()
    except KeyboardInterrupt:
        os._exit(0)

if __name__ == "__main__":
    main()

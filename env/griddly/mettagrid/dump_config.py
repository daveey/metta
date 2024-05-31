import hydra
from omegaconf import OmegaConf
import yaml

@hydra.main(version_base=None, config_path="../../../configs", config_name="configs")
def main(cfg):
    print(OmegaConf.to_yaml(cfg))
    builder = hydra.utils.instantiate(cfg.env.env_builder)
    griddly_yaml = builder.build()

    config_path = "/tmp/griddly_env.yaml"
    with open(config_path, "w") as f:
        yaml.dump(griddly_yaml, f, sort_keys=False)
    print(f"Griddly environment config saved to {config_path}")

if __name__ == "__main__":
    main()

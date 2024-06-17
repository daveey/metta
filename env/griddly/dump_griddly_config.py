import hydra
from omegaconf import OmegaConf
import yaml

from env.griddly.mettagrid.game_builder import MettaGridGameBuilder
from util.sample_config import sample_config


@hydra.main(version_base=None, config_path="../../configs", config_name="configs")
def main(cfg):
    print(OmegaConf.to_yaml(cfg))

    builder = MettaGridGameBuilder(**sample_config(cfg.env.game))
    griddly_yaml = builder.build()

    config_path = "/tmp/griddly_env.yaml"
    with open(config_path, "w") as f:
        f.write(griddly_yaml)
        print(f"Griddly environment config saved to {config_path}")

if __name__ == "__main__":
    main()

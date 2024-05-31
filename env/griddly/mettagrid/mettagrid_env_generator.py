
import numpy as np
from omegaconf import OmegaConf
import yaml

import jmespath
from env.griddly.mettagrid.objects import MettaGridBuilder

class MettaGridEnvGenerator():
    def __init__(self, **cfg):
        cfg = OmegaConf.create(cfg)
        self._cfg = cfg

        with open("./env/griddly/mettagrid/gdy/mettagrid_gdy.yaml", encoding="utf-8") as file:
            self._griddly_config_template = yaml.safe_load(file)

    def generate_level_config(self):
        return self._cfg

    def generate_config(self):
        level_cfg = self.generate_level_config()
        game = MettaGridBuilder(self._griddly_config_template, level_cfg)
        griddly_cfg = game.build()
        return level_cfg, griddly_cfg


    def _generate_level(self, level_cfg):
        width = level_cfg.width
        height = level_cfg.height

        level = np.array([["."] * width] * height).astype("U6") # 2-char unicode strings
        floor_tiles = [".", "o"]

        # make the bounding box
        level[0,:] = "W"
        level[-1,:] = "W"
        level[:,0] = "W"
        level[:,-1] = "W"

        # make the agents
        for i in range(level_cfg.agents.count):
            while True:
                x = np.random.randint(1, width-1)
                y = np.random.randint(1, height-1)
                if level[y][x] in floor_tiles:
                    level[y][x] = f"A{i+1}"
                    break

        # make the objects:
        # for obj_name, obj_cfg in level_cfg.objects.items():
        #     if "count" in obj_cfg:
        #         count = obj_cfg.count
        #     else:
        #         count = int(obj_cfg.density * width * height)
        #     obj = OBJECTS[obj_name](obj_name, obj_cfg)

        #     for i in range(count):
        #         for _ in range(10):
        #             x = np.random.randint(1, width-1)
        #             y = np.random.randint(1, height-1)
        #             if level[y][x] in floor_tiles:
        #                 level[y][x] = obj.symbol
        #                 break

        return level

    def sample_cfg(self, vals):
        if isinstance(vals, (int, float)):
            return vals
        if len(vals) == 1:
            return vals[0]
        elif len(vals) == 2:
            return np.random.uniform(vals[0], vals[1])
        raise ValueError(f"Length of values list should be at most 2. Got: {len(vals)}")


# def _update_global_variable(game_config, var_name, value):
#     jmespath.search('Environment.Variables[?Name==`{}`][]'.format(var_name), game_config)[0]["InitialValue"] = value

# def _update_object_variable(game_config, object_name, var_name, value):
#     jmespath.search(
#         f'Objects[?Name==`{object_name}`][].Variables[?Name==`{var_name}`][]',
#         game_config)[0]["InitialValue"] = value

# def _update_object_count(game_config, object_name, count):
#     for i in range(self._cfg.get("extra_variables", 0)):
#     jmespath.search(
#         f'Objects[?Name==`agent`][].Variables', game_config)[0].append({
#             "Name": f"agent:extra_property:{i}",
#             "InitialValue": 0
#         })


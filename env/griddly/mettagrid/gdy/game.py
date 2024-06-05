import numpy as np
from omegaconf import OmegaConf

from env.griddly.builder.game import GriddlyGame
from env.griddly.mettagrid.gdy.actions.attack import Attack
from env.griddly.mettagrid.gdy.actions.move import Move
from env.griddly.mettagrid.gdy.actions.transfer import Transfer
from env.griddly.mettagrid.gdy.actions.rotate import Rotate
from env.griddly.mettagrid.gdy.actions.shield import Shield
from env.griddly.mettagrid.gdy.actions.use import Use
from env.griddly.mettagrid.gdy.objects.agent import Agent
from env.griddly.mettagrid.gdy.objects.altar import Altar
from env.griddly.mettagrid.gdy.objects.converter import Converter
from env.griddly.mettagrid.gdy.objects.generator import Generator
from env.griddly.mettagrid.gdy.objects.wall import Wall

class MettaGrid(GriddlyGame):
    def __init__(
            self,
            obs_width: int,
            obs_height: int,
            tile_size: int,
            width: int,
            height: int,
            max_steps: int,
            objects,
            actions):

        super().__init__(
            obs_width=obs_width,
            obs_height=obs_height,
            tile_size=tile_size,
            width=width,
            height=height,
            num_agents=objects.agent.count,
            max_steps=max_steps
        )
        objects = OmegaConf.create(objects)
        self.object_configs = objects
        actions = OmegaConf.create(actions)
        self.action_configs = actions

        self.register_object(Agent(self, objects.agent))
        self.register_object(Altar(self, objects.altar))
        self.register_object(Converter(self, objects.converter))
        self.register_object(Generator(self, objects.generator))
        self.register_object(Wall(self, objects.wall))

        self.register_action(Move(self, actions.move))
        self.register_action(Rotate(self, actions.rotate))
        self.register_action(Use(self, actions.use))
        self.register_action(Transfer(self, actions.drop))
        self.register_action(Attack(self, actions.attack))
        self.register_action(Shield(self, actions.shield))

    def level(self):
        level = np.array([["."] * self.width] * self.height).astype("U6")
        floor_tiles = [".", "o"]

        level[0,:] = "W"
        level[-1,:] = "W"
        level[:,0] = "W"
        level[:,-1] = "W"

        for i in range(self.num_agents):
            while True:
                x = np.random.randint(1, self.width-1)
                y = np.random.randint(1, self.height-1)
                if level[y][x] in floor_tiles:
                    level[y][x] = f"A{i+1}"
                    break

        for obj in self.objects():
            if obj.name == "agent":
                continue

            obj_config = self.object_configs[obj.name]
            if "count" in obj_config:
                count = obj_config.count
            else:
                count = int(obj_config.density * self.width * self.height)

            for i in range(count):
                for _ in range(10):
                    x = np.random.randint(1, self.width-1)
                    y = np.random.randint(1, self.height-1)
                    if level[y][x] in floor_tiles:
                        level[y][x] = obj.symbol
                        break
        return level

        # level = self._generate_level(self.level_cfg)
        # griddly_cfg["Environment"]["Levels"] = ["\n".join(["  ".join(row) for row in level])]
        # for name, obj_cfg in level_cfg.objects.items():
        #     obj = OBJECTS[name](name, obj_cfg)
        #     griddly_cfg["Environment"]["Variables"].extend(obj.gen_global_vars())
        #     griddly_cfg["Objects"].append(obj.gen_object())

        # for name, action in level_cfg.actions.items():
        #     for k, v in action.items():
        #         griddly_cfg["Environment"]["Variables"].append({
        #             "Name": f"conf:action:{name}:{k}",
        #             "InitialValue": v,
        #             "PerPlayer": True
        #         })
        #     griddly_cfg["Environment"]["Variables"].append({
        #         "Name": f"stats:action:{name}",
        #         "InitialValue": 0,
        #         "PerPlayer": True
        #     })
        #     griddly_cfg["Environment"]["Variables"].append({
        #         "Name": f"stats:energy:action:{name}",
        #         "InitialValue": 0,
        #         "PerPlayer": True
        #     })

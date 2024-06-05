import numpy as np
from omegaconf import OmegaConf

from env.griddly.builder.game_builder import GriddlyGameBuilder
from env.griddly.mettagrid.action.attack import Attack
from env.griddly.mettagrid.action.move import Move
from env.griddly.mettagrid.action.transfer import Transfer
from env.griddly.mettagrid.action.rotate import Rotate
from env.griddly.mettagrid.action.shield import Shield
from env.griddly.mettagrid.action.use import Use
from env.griddly.mettagrid.object.agent import Agent
from env.griddly.mettagrid.object.altar import Altar
from env.griddly.mettagrid.object.converter import Converter
from env.griddly.mettagrid.object.generator import Generator
from env.griddly.mettagrid.object.wall import Wall

class MettaGridGameBuilder(GriddlyGameBuilder):
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


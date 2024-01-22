from typing import Optional
from griddly.gym import GymWrapper
import numpy as np
import yaml

GYM_ENV_NAME = "GDY-Forage"

class ForageEnvFactory:
    def __init__(self, cfg=None):
        self.cfg = cfg
        self.game_config = yaml.safe_load(open("./envs/griddly/forage/forage.yaml"))
        self.num_agents = int(self.game_config["Environment"]["Player"]["Count"])

    def make(self):
        return GymWrapper(
            yaml_string=yaml.dump(self.game_config),
            player_observer_type="VectorAgent",
            global_observer_type="GlobalSpriteObserver",
            level=0,
            max_steps=1024
        )

    def make_level_string(self):
        width = height = np.random.randint(10, 50)
        width = height = 20
        # height = np.random.randint(10, 50)
        level = self._make_level(
            width=width, height=height,
            num_agents=self.num_agents,
            num_food=self.num_agents*10
        )
        return "\n".join(["  ".join(row) for row in level])

    def _make_level(self,
                    width: int = 10,
                    height: int = 10,
                    num_agents: int = 1,
                    num_food: int = 1):

        # make the bounding box
        level = [["."] * width for _ in range(height)]
        level[0] = ["W"] * width
        level[-1] = ["W"] * width
        for i in range(height):
            level[i][0] = "W"
            level[i][-1] = "W"

        # make the agents
        for i in range(num_agents):
            while True:
                x = np.random.randint(1, width-1)
                y = np.random.randint(1, height-1)
                if level[y][x] == ".":
                    level[y][x] = f"A{i+1}"
                    break

        # make the energy
        for i in range(num_food):
            for _ in range(10):
                x = np.random.randint(1, width-1)
                y = np.random.randint(1, height-1)
                if level[y][x] == ".":
                    level[y][x] = "e"
                    break

        # make obstacles
        for i in range(width*height//10):
            x = np.random.randint(1, width-1)
            y = np.random.randint(1, height-1)
            if level[y][x] == ".":
                level[y][x] = "W"
        return level

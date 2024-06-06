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
            max_steps: int,
            num_agents: int,
            objects,
            actions,
            map):

        super().__init__(
            obs_width=obs_width,
            obs_height=obs_height,
            tile_size=tile_size,
            num_agents=num_agents,
            max_steps=max_steps
        )
        objects = OmegaConf.create(objects)
        self.object_configs = objects
        actions = OmegaConf.create(actions)
        self.action_configs = actions
        self.map_config = OmegaConf.create(map)

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
        num_agents = 0
        layers = []
        for layer in self.map_config.layout:
            rooms = []
            for room_name in layer:
                room_config = self.map_config[room_name]
                rooms.append(self.build_room(room_config, num_agents + 1))
                num_agents += room_config.objects.agent
            layers.append(np.concatenate(rooms, axis=1))
        level = np.concatenate(layers, axis=0)
        assert num_agents == self.num_agents, f"Number of agents in map ({num_agents}) does not match num_agents ({self.num_agents})"
        return level


    def build_room(self, room_config, starting_agent=1):
        symbols = []
        content_width = room_config.width - 2*room_config.border
        content_height = room_config.height - 2*room_config.border
        area = content_width * content_height

        for obj_name, count in room_config.objects.items():
            symbol = self._objects[obj_name].symbol
            if obj_name == "agent":
                symbols.extend([f"{symbol}{i+starting_agent}" for i in range(count)])
            else:
                symbols.extend([symbol] * count)

        assert(len(symbols) <= area), f"Too many objects in room: {len(symbols)} > {area}"
        symbols.extend(["."] * (area - len(symbols)))
        symbols = np.array(symbols).astype("U6")
        np.random.shuffle(symbols)
        content = symbols.reshape(content_height, content_width)
        room = np.full((room_config.height, room_config.width), "W", dtype="U6")
        room[room_config.border:room_config.border+content_height,
             room_config.border:room_config.border+content_width] = content

        return room


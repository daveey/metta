
from libc.stdio cimport printf

import numpy as np
cimport numpy as cnp

from puffergrid.grid_env cimport GridEnv
from puffergrid.action cimport ActionHandler
from omegaconf import OmegaConf
from env.mettagrid.objects cimport ObjectType, Agent, Wall, GridLayer, Generator, Converter, Altar
from env.mettagrid.objects cimport MettaObservationEncoder
from env.mettagrid.actions cimport Move, Rotate, Use, Attack, ToggleShield, Gift
from puffergrid.action cimport ActionHandler, ActionArg


ObjectLayers = {
    ObjectType.AgentT: GridLayer.Agent_Layer,
    ObjectType.WallT: GridLayer.Object_Layer,
    ObjectType.GeneratorT: GridLayer.Object_Layer,
    ObjectType.ConverterT: GridLayer.Object_Layer,
    ObjectType.AltarT: GridLayer.Object_Layer,
}

cdef class MettaGrid(GridEnv):
    cdef:
        object _cfg

    def __init__(self, cfg: OmegaConf, map: np.ndarray):
        self._cfg = cfg

        GridEnv.__init__(
            self,
            cfg.num_agents,
            map.shape[1],
            map.shape[0],
            0, # max_steps
            ObjectLayers.values(),
            11,11,
            MettaObservationEncoder(),
            [
                Move(),
                Rotate(),
                Use(),
                Attack(),
                ToggleShield(),
                Gift(),
            ],
            [
            ]
        )


        cdef Agent *agent
        cdef Wall *wall
        cdef Altar *altar

        for r in range(map.shape[0]):
            for c in range(map.shape[1]):
                if map[r,c] == "W":
                    wall = new Wall(r, c)
                    self._grid.add_object(wall)
                    self._stats.game_incr("objects.wall")
                elif map[r,c] == "g":
                    self._grid.add_object(new Generator(r, c))
                    self._stats.game_incr("objects.generator")
                elif map[r,c] == "c":
                    self._grid.add_object(new Converter(r, c))
                    self._stats.game_incr("objects.converter")
                elif map[r,c] == "a":
                    self._grid.add_object(new Altar(r, c))
                    self._stats.game_incr("objects.altar")
                elif map[r,c][0] == "A":
                    agent = new Agent(r, c)
                    self._grid.add_object(agent)
                    self.add_agent(agent)
                    self._stats.game_incr("objects.agent")
                    print("agent", r, c)


    def render(self):
        grid = self.render_ascii(["A", "#", "g", "c", "a"])
        for r in grid:
            print("".join(r))

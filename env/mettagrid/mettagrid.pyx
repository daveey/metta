
from libc.stdio cimport printf

import numpy as np
cimport numpy as cnp

from puffergrid.grid_env cimport GridEnv
from puffergrid.action cimport ActionHandler
from omegaconf import OmegaConf
from env.mettagrid.objects cimport ObjectLayers, Agent, ResetHandler, Wall, GridLayer, Generator, Converter, Altar
from env.mettagrid.objects cimport MettaObservationEncoder
from puffergrid.action cimport ActionHandler, ActionArg
from env.mettagrid.actions.move import Move
from env.mettagrid.actions.rotate import Rotate
from env.mettagrid.actions.use import Use
from env.mettagrid.actions.attack import Attack
from env.mettagrid.actions.shield import Shield
from env.mettagrid.actions.gift import Gift

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
            dict(ObjectLayers).values(),
            11,11,
            MettaObservationEncoder(),
            [
                Move(),
                Rotate(),
                Use(),
                Attack(),
                Shield(),
                Gift(),
            ],
            [
                ResetHandler()
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

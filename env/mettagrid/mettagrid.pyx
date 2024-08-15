
from libc.stdio cimport printf

import numpy as np

from omegaconf import OmegaConf

from puffergrid.grid_env cimport GridEnv
from env.mettagrid.objects cimport ObjectLayers, Agent, ResetHandler, Wall, Generator, Converter, Altar
from env.mettagrid.objects cimport MettaObservationEncoder
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
            cfg.max_steps,
            dict(ObjectLayers).values(),
            cfg.obs_width, cfg.obs_height,
            MettaObservationEncoder(),
            [
                Move(cfg.actions.move),
                Rotate(cfg.actions.rotate),
                Use(cfg.actions.use),
                Attack(cfg.actions.attack),
                Shield(cfg.actions.shield),
                Gift(cfg.actions.gift),
            ],
            [
                ResetHandler()
            ]
        )


        cdef Agent *agent
        for r in range(map.shape[0]):
            for c in range(map.shape[1]):
                if map[r,c] == "W":
                    self._grid.add_object(new Wall(r, c, cfg.objects.wall))
                    self._stats.game_incr("objects.wall")
                elif map[r,c] == "g":
                    self._grid.add_object(new Generator(r, c, cfg.objects.generator))
                    self._stats.game_incr("objects.generator")
                elif map[r,c] == "c":
                    self._grid.add_object(new Converter(r, c, cfg.objects.converter))
                    self._stats.game_incr("objects.converter")
                elif map[r,c] == "a":
                    self._grid.add_object(new Altar(r, c, cfg.objects.altar))
                    self._stats.game_incr("objects.altar")
                elif map[r,c][0] == "A":
                    agent = new Agent(r, c, cfg.objects.agent)
                    self._grid.add_object(agent)
                    self.add_agent(agent)
                    self._stats.game_incr("objects.agent")


    def render(self):
        grid = self.render_ascii(["A", "#", "g", "c", "a"])
        for r in grid:
            print("".join(r))

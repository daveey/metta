
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
        agent = self._grid.create_object[Agent](ObjectType.AltarT, 4, 6)

        self._grid.create_object[Altar](ObjectType.AltarT, 5, 5)
        self._grid.create_object[Altar](ObjectType.AltarT, 5, 6)
        self._grid.create_object[Altar](ObjectType.AltarT, 5, 7)
        self._grid.create_object[Altar](ObjectType.AltarT, 3, 5)
        self._grid.create_object[Altar](ObjectType.AltarT, 3, 6)
        self._grid.create_object[Altar](ObjectType.AltarT, 3, 7)
        self._grid.create_object[Altar](ObjectType.AltarT, 4, 5)
        self._grid.create_object[Altar](ObjectType.AltarT, 4, 7)
        self.add_agent(agent)

        return
        for r in range(map.shape[0]):
            for c in range(map.shape[1]):
                if map[r,c] == "W":
                    wall = self._grid.create_object[Wall](ObjectType.WallT, r, c)
                    self._stats.game_incr("objects.wall")
                    wall.props.hp = 10
                elif map[r,c] == "g":
                    self._grid.create_object[Generator](ObjectType.GeneratorT, r, c)
                    self._stats.game_incr("objects.generator")
                elif map[r,c] == "c":
                    self._grid.create_object[Converter](ObjectType.ConverterT, r, c)
                    self._stats.game_incr("objects.converter")
                elif map[r,c] == "a":
                    altar = self._grid.create_object[Altar](ObjectType.AltarT, r, c)
                    self._stats.game_incr("objects.altar")
                    altar.props.hp = 100
                    altar.props.ready = 1
                    altar.props.energy_cost = 100
                elif map[r,c][0] == "A":
                    agent = self._grid.create_object[Agent](ObjectType.AgentT, r, c)
                    agent.props.hp = 111
                    agent.props.energy = 500
                    self.add_agent(agent)
                    self._stats.game_incr("objects.agent")
                    print("agent", r, c)


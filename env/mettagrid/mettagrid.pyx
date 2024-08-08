
from libc.stdio cimport printf

import numpy as np
cimport numpy as cnp

from puffergrid.grid_env cimport GridEnv
from puffergrid.action cimport ActionHandler
from omegaconf import OmegaConf
from env.mettagrid.objects cimport ObjectType, Agent, Wall, GridLayer_Agent, GridLayer_Object
from env.mettagrid.objects cimport MettaObservationEncoder
from env.mettagrid.actions cimport Move, Rotate, Use, Attack, ToggleShield, Gift
from puffergrid.action cimport ActionHandler, ActionArg


ObjectLayers = {
    ObjectType.AgentT: GridLayer_Agent,
    ObjectType.WallT: GridLayer_Object,
    ObjectType.GeneratorT: GridLayer_Object,
    ObjectType.ConverterT: GridLayer_Object,
    ObjectType.AltarT: GridLayer_Object,
}

cdef class MettaGrid(GridEnv):
    cdef:
        object _cfg

    def __init__(self, cfg: OmegaConf, map: np.ndarray):
        self._cfg = cfg

        GridEnv.__init__(
            self,
            cfg.num_agents,
            map.shape[0],
            map.shape[1],
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


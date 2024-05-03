
from meltingpot import substrate

from env.meltingpot.melting_pot_env import MeltingPotEnv

class HarvestEnv(MeltingPotEnv):
    def __init__(self, env_id:int=0, **cfg):
        self._cfg = cfg
        env_name = "allelopathic_harvest__open"
        env_config = substrate.get_config(env_name)
        super().__init__(env_config, max_cycles=1000)

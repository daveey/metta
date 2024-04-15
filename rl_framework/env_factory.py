from typing import Optional


from envs.griddly.power_grid import power_grid_env, power_grid_level_generator
from envs.griddly.sample_factory_env_wrapper import GriddlyEnvWrapper

class EnvFactory():
    def __init__(self, cfg):
        self.cfg = cfg

    def make_env(self, env_id: int = 0, render_mode: Optional[str] = None):
        lg = power_grid_level_generator.PowerGridLevelGenerator(self.cfg)
        env = power_grid_env.PowerGridEnv(lg, render_mode=render_mode)
        return GriddlyEnvWrapper(
            env,
            render_mode=render_mode,
            make_level=lg.make_level_string,
            env_id=env_id,
        )

    def gym_env_name(self):
        return power_grid_env.GYM_ENV_NAME


from griddly.wrappers.render_wrapper import RenderWrapper
import gymnasium as gym

class GriddlyEnv(gym.Env):
    def __init__(self, render_mode, **cfg):
        self._cfg = cfg
        self._global_env = RenderWrapper(
            self, "global",
            render_mode=render_mode
        )

    def render(self, *args, **kwargs):
        return self._global_env.render()
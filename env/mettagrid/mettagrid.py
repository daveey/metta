
from types import SimpleNamespace
import numpy as np
from env.puffergrid.env import PufferGridEnv
from pufferlib.environment import PufferEnv


from env.mettagrid.mettagrid_c import MettaGrid

if __name__ == '__main__':
    env = PufferGridEnv(
        MettaGrid,
        map_width=40,
        map_height=40,
        num_agents=5,
        max_timesteps=1000,
        obs_width=11,
        obs_height=11)
    print("success")

    # actions = [{i+1: e for i, e in enumerate(np.random.randint(0, 5, env.num_agents))}
    #     for i in range(1000)]
    # idx = 0
    # dones = {1: True}
    # start = time.time()
    # while True:
    #     if all(dones.values()):
    #         env.reset()
    #         dones = {1: False}
    #     else:
    #         _, _, dones, _, _ = env.step(actions[idx%1000])

    #     idx += 1
    #     frame = env.render()
    #     import cv2
    #     frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    #     cv2.imshow('frame', frame)
    #     cv2.waitKey(1)

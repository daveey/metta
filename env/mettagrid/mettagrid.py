
from math import e
import time
from types import SimpleNamespace
import numpy as np
from torch import rand
from env.puffergrid.grid_env import PufferGridEnv
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

    for c in range(0, env._map_width):
        if env.grid_location_empty(0, c):
            env.add_object(env.type_ids.Wall, 0, c)
        if env.grid_location_empty(env._map_height-1, c):
            env.add_object(env.type_ids.Wall, env._map_height-1, c)

    for r in range(0, env._map_height):
        if env.grid_location_empty(r, 0):
            env.add_object(env.type_ids.Wall, r, 0)
        if env.grid_location_empty(r, env._map_width-1):
            env.add_object(env.type_ids.Wall, r, env._map_width-1)

    for agent_id in range(env._num_agents):
        while True:
            c = np.random.randint(0, env._map_width)
            r = np.random.randint(0, env._map_height)
            if env.grid_location_empty(r, c):
                agent = env.add_agent(
                    env.type_ids.Agent, r, c,
                    id=agent_id
                )
                print("adding agent", agent_id, "at", r, c, "id:", agent)
                break

    for tree in range(50):
        r = np.random.randint(0, env._map_width)
        c = np.random.randint(0, env._map_height)
        if env.grid_location_empty(r, c):
            tree = env.add_object(env.type_ids.Tree, r, c)

    start = time.time()
    obs, infos = env.reset(0)
    while True:
        env._buffers.actions[:] = np.random.randint(0, 10, (env._num_agents, 2), dtype=np.uint32)
        obs, rewards, terms, truncs, infos = env.step(env._buffers.actions[:])
        if rewards.sum() > 0:
            print("rewards", rewards)
        if env.current_timestep % 10000 == 0:
            print("timestep", env.current_timestep)
            print("SPS: ", env.current_timestep,
                  env.current_timestep / (time.time() - start))

        if env.current_timestep > 1000000:
            break

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

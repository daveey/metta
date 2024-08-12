import time

import hydra
import numpy as np
from tqdm import tqdm

global actions
global env

def test_performance(env, actions, duration):
    tick = 0
    num_actions = actions.shape[0]
    start = time.time()
    with tqdm(total=duration, desc="Running performance test") as pbar:
        while time.time() - start < duration:
            atns = actions[tick % num_actions]
            env.step(atns)
            tick += 1
            if tick % 100 == 0:
                pbar.update(time.time() - start - pbar.n)

    print(env._c_env.stats())
    print(f'SPS: {atns.shape[0] * tick / (time.time() - start):.2f}')

actions = {}
env = {}
@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg):
    # Run with c profile
    from cProfile import run
    global env

    cfg.env.game.max_steps = np.inf
    env = hydra.utils.instantiate(cfg.env, render_mode="human")
    env.reset()
    global actions
    num_agents = cfg.env.game.num_agents
    actions = np.random.randint(0, env.action_space.nvec, (1024, num_agents, 2), dtype=np.uint32)

    test_performance(env, actions, 5)
    exit(0)

    run("test_performance(env, actions, 10)", 'stats.profile')
    import pstats
    from pstats import SortKey
    p = pstats.Stats('stats.profile')
    p.sort_stats(SortKey.TIME).print_stats(25)
    exit(0)


if __name__ == "__main__":
    main()

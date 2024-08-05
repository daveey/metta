import hydra
import numpy as np

def test_performance(
        env,
        actions,
        timeout=20,
        atn_cache=1024,
        num_envs=400):
    tick = 0

    import time
    start = time.time()
    while time.time() - start < timeout:
        atns = actions[tick % atn_cache]
        env.step(atns)
        tick += 1

    print(f'SPS: %f', atns.shape[0]*num_envs*tick / (time.time() - start))

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
    actions = np.random.randint(0, env.action_space.nvec, (1024, cfg.env.game.num_agents, 2))

    test_performance(env, actions, num_agents, 1024, 1)
    exit(0)

    run(f"test_performance(env, actions, {num_agents}, 1024, 1)",
         'stats.profile')
    import pstats
    from pstats import SortKey
    p = pstats.Stats('stats.profile')
    p.sort_stats(SortKey.TIME).print_stats(25)
    exit(0)


if __name__ == "__main__":
    main()

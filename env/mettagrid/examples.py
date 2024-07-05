
def test_pz_performance(timeout):
    import time
    env = PettingZooGrid()
    actions = [{i+1: e for i, e in enumerate(np.random.randint(0, 5, env.num_agents))}
        for i in range(1000)]
    idx = 0
    dones = {1: True}
    start = time.time()
    while time.time() - start < timeout:
        if all(dones.values()):
            env.reset()
            dones = {1: False}
        else:
            _, _, dones, _, _ = env.step(actions[idx%1000])

        idx += 1

    sps = env.num_agents * idx // timeout
    print(f'PZ SPS: {sps}')

def test_puffer_performance(timeout):
    import time
    env = PufferGrid()
    actions = np.random.randint(0, 5, (1000, env.num_agents))
    idx = 0
    dones = {1: True}
    start = time.time()
    while time.time() - start < timeout:
        if env.done:
            env.reset()
            dones = {1: False}
        else:
            _, _, dones, _, _ = env.step(actions[idx%1000])

        idx += 1

    sps = env.num_agents * idx // timeout
    print(f'Puffer SPS: {sps}')


class PettingZooGrid(pettingzoo.ParallelEnv):
    def __init__(self, map_size=80, num_agents=3, horizon=1024, vision_range=10, render_mode='human'):
        super().__init__()
        self.env = PufferGrid(map_size, 2, num_agents, horizon, vision_range, render_mode)
        self.possible_agents = [i+1 for i in range(num_agents)]
        self.infos = {i: {} for i in self.possible_agents}
        self.observation_space = observation_space
        self.action_space = action_space

    @property
    def num_agents(self):
        return self.env.num_agents

    def reset(self, seed=0):
        obs, _ = self.env.reset(seed)
        observations = {i+1: obs[i] for i in range(self.num_agents)}
        return observations, self.infos

    def step(self, actions):
        actions = np.array(list(actions.values()), dtype=np.uint32)
        obs, rewards, terminals, truncations, infos = self.env.step(actions)
        observations = {i+1: obs[i] for i in range(self.num_agents)}
        rewards = {i+1: rewards[i] for i in range(self.num_agents)}
        terminals = {i+1: terminals[i] for i in range(self.num_agents)}
        truncations = {i+1: truncations[i] for i in range(self.num_agents)}
        return observations, rewards, terminals, truncations, self.infos

    def render(self):
        return self.env.render()

if __name__ == '__main__':
    test_random_actions(10)
    # test_puffer_performance(10)
    # test_pz_performance(10)
    # use_c = False
    # test_puffer_performance(10)
    # test_pz_performance(10)

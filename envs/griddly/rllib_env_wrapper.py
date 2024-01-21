class RLlibMultiAgentWrapper(gym.Wrapper, MultiAgentEnv):
    def __init__(self, env, env_config):
        super().__init__(env)

        self._player_done_variable = env_config.get("player_done_variable", None)

        # Used to keep track of agents that are active in the environment
        self._active_agents = set()

        self._agent_recorders = None
        self._global_recorder = None

        self._worker_idx = None
        self._env_idx = None

        assert (
            self.player_count > 1
        ), "RLlibMultiAgentWrapper can only be used with environments that have multiple agents"

    def _to_multi_agent_map(self, data):
        return {a: data[a - 1] for a in self._active_agents}

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        self._active_agents.update([a + 1 for a in range(self.player_count)])
        return self._to_multi_agent_map(obs)

    def _resolve_player_done_variable(self):
        resolved_variables = self.game.get_global_variable([self._player_done_variable])
        return resolved_variables[self._player_done_variable]

    def _after_step(self, obs_map, reward_map, done_map, info_map):
        extra_info = {}

        if self.is_video_enabled():
            videos_list = []
            if self.include_agent_videos:
                for a in self._active_agents:
                    video_info = self._agent_recorders[a].step(
                        self.level_id, self.env_steps, done_map[a - 1]
                    )
                    if video_info is not None:
                        videos_list.append(video_info)
            if self.include_global_video:
                video_info = self._global_recorder.step(
                    self.level_id, self.env_steps, done_map["__all__"]
                )
                if video_info is not None:
                    videos_list.append(video_info)

            self.videos = videos_list

        return extra_info

    def step(self, action_dict: MultiAgentDict):
        actions_array = [None] * self.player_count
        for agent_id, action in action_dict.items():
            actions_array[agent_id - 1] = action

        obs, reward, all_done, info = super().step(actions_array)

        done_map = {"__all__": all_done}

        if self._player_done_variable is not None:
            griddly_players_done = self._resolve_player_done_variable()

            for agent_id in self._active_agents:
                done_map[agent_id] = griddly_players_done[agent_id] == 1 or all_done
        else:
            for p in range(self.player_count):
                done_map[p] = False

        if self.generate_valid_action_trees:
            info_map = self._to_multi_agent_map(
                [
                    {"valid_action_tree": valid_action_tree}
                    for valid_action_tree in info["valid_action_tree"]
                ]
            )
        else:
            info_map = self._to_multi_agent_map(defaultdict(dict))

        if self.record_actions:
            for event in info["History"]:
                event_player_id = event["PlayerId"]
                if event_player_id != 0:
                    if "History" not in info_map[event_player_id]:
                        info_map[event_player_id]["History"] = []
                    info_map[event_player_id]["History"].append(event)

        obs_map = self._to_multi_agent_map(obs)
        reward_map = self._to_multi_agent_map(reward)

        # Finally remove any agent ids that are done
        for agent_id, is_done in done_map.items():
            if is_done:
                self._active_agents.discard(agent_id)

        self._after_step(obs_map, reward_map, done_map, info_map)

        assert len(obs_map) == len(reward_map)
        assert len(obs_map) == len(done_map) - 1
        assert len(obs_map) == len(info_map)

        return obs_map, reward_map, done_map, info_map

    def is_video_enabled(self):
        return (
            self.record_video_config is not None
            and self._env_idx is not None
            and self._env_idx == 0
        )

    def on_episode_start(self, worker_idx, env_idx):
        self._env_idx = env_idx
        self._worker_idx = worker_idx

        if self.is_video_enabled() and not self.video_initialized:
            self.init_video_recording()
            self.video_initialized = True

    def init_video_recording(self):
        if self.include_agent_videos:
            self._agent_recorders = {}
            for a in range(self.player_count):
                agent_id = a + 1
                self._agent_recorders[agent_id] = ObserverEpisodeRecorder(
                    self, agent_id, self.video_frequency, self.video_directory
                )
        if self.include_global_video:
            self._global_recorder = ObserverEpisodeRecorder(
                self, "global", self.video_frequency, self.video_directory
            )

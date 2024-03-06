
def load_scenario_config(cfg):
    if cfg.scenario is None:
        return

    if cfg.scenario == "metabolism":
        from scenario.metabolism import training_config
        cfg.update(training_config)

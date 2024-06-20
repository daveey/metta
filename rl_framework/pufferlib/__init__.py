from .puffer_env_wrapper import env_creator, make

try:
    import rl_framework.pufferlib.puffer_agent_wrapper as puffer_agent_wrapper
except ImportError:
    pass
else:
    from .puffer_agent_wrapper import Policy
    try:
        from .puffer_agent_wrapper import Recurrent
    except:
        Recurrent = None

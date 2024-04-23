from omegaconf import OmegaConf
from sample_factory.algo.utils.context import global_model_factory
from sample_factory.model.model_factory import MakeEncoderFunc
from sample_factory.utils.typing import Config, ObsSpace
from agent.predicting_actor_critic import make_actor_critic_func

class SampleFactoryAgent():
    def __init__(self, *args, **kwargs):
        self._cfg = OmegaConf.structured(kwargs)

        self._global_model_factory = global_model_factory()
        self._global_model_factory.register_encoder_factory(self.make_encoder)
        self._global_model_factory.register_decoder_factory(self.make_decoder)
        self._global_model_factory.register_actor_critic_factory(make_actor_critic_func)

    def make_encoder(self, cfg: Config, obs_space: ObsSpace):
        return self.encoder_cls()(cfg, obs_space, self._cfg)

    def make_decoder(self, cfg: Config, size: int):
        return self.decoder_cls()(cfg, size)

    def encoder_cls(self):
        raise NotImplementedError

    def decoder_cls(self):
        raise NotImplementedError


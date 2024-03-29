from sample_factory.algo.utils.context import global_model_factory
from agent.predicting_actor_critic import make_actor_critic_func

class SampleFactoryAgent():
    def __init__(self):
        self._global_model_factory = global_model_factory()
        self._global_model_factory.register_encoder_factory(self.encoder_cls())
        self._global_model_factory.register_decoder_factory(self.decoder_cls())
        self._global_model_factory.register_actor_critic_factory(make_actor_critic_func)

    def encoder_cls(self):
        raise NotImplementedError()

    def decoder_cls(self):
        raise NotImplementedError()

    def add_args(self, parser):
        pass


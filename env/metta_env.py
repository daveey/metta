import gymnasium as gym

class FeatureSchemaInterface:
    def feature_schema(self):
        raise NotImplementedError

class RewardSharingInterface:
    def reward_sharing(self):
        raise NotImplementedError

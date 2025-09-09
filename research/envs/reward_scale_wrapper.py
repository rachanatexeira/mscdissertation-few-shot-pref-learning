import gymnasium as gym

class RewardScaleWrapper(gym.RewardWrapper):
    def __init__(self, env, scale=0.1):
        super().__init__(env)
        self.scale = float(scale)

    def reward(self, r):
        return self.scale * r
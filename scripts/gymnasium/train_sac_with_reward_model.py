import argparse
import torch
import numpy as np
import gymnasium as gym
import gymnasium_robotics

from research.utils.trainer_gym import get_model
from research.utils.config import Config

from research.utils.trainer_gym import load_from_path

from research.envs.custom_fetch_env import CustomFetchReach
from research.networks import ActorCriticPolicy
from research.algs.sac import SAC
from research.datasets import ReplayBuffer
import research.envs  

from research.utils.train_reward_model import RewardModel  # adjust path if needed


from gymnasium.wrappers import FlattenObservation


class LearnedRewardWrapper(gym.RewardWrapper):
    """
    A Gym wrapper that replaces the environment's reward signal
    with the learned reward model predictions.
    """
    def __init__(self, env, reward_model_path, obs_dim):
        super().__init__(env)
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.reward_model = RewardModel(obs_dim).to(self.device)
        state_dict = torch.load(reward_model_path, map_location=self.device)
        self.reward_model.load_state_dict(state_dict)
        self.reward_model.eval()
        self.obs_dim = obs_dim

    def reward(self, reward):
        # Ignore env reward — use learned reward model
        obs = self.env.unwrapped._get_obs()  # FetchReach style
        obs = np.array(obs, dtype=np.float32)
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

        with torch.no_grad():
            predicted_reward = self.reward_model(obs_tensor)

        return predicted_reward.item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to SAC config file")
    parser.add_argument("--reward-model", type=str, required=True, help="Path to trained reward model (.pt)")
    parser.add_argument("--save-path", type=str, required=True, help="Where to save final preference-trained SAC model")
    args = parser.parse_args()

    # Load config
    config = Config.load(args.config)
    config = config.parse()

    # Create environment
    env = gym.make(config["env"], **config["env_kwargs"])
    env = FlattenObservation(env) 

    # Replace reward with learned reward
    obs_dim = env.observation_space.shape[0]
    env = LearnedRewardWrapper(env, args.reward_model, obs_dim)

    # Load SAC model from config
    model = get_model(config)

    print("[INFO] Training SAC agent using learned reward model...")
    model.train(
        args.save_path,
        **config["train_kwargs"]
    )

    print(f"[INFO] Training complete. Model saved to: {args.save_path}")


if __name__ == "__main__":
    main()
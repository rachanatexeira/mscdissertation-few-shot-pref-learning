import os
import argparse
import numpy as np
import gymnasium as gym
import gymnasium_robotics
import torch
import yaml
from research.networks import ActorCriticPolicy
from research.algs.sac import SAC
from research.datasets import ReplayBuffer
import research.envs   
 

from gymnasium.wrappers import FlattenObservation




def load_trained_policy(env, policy_path, config_path):
    """
    Rebuild the SAC policy architecture and load trained weights.
    """
    # 1. Load the YAML config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 2. Build SAC using the correct network kwargs
    policy = SAC(
        env=env,
        network_class=lambda obs_space, act_space: ActorCriticPolicy(
            obs_space,
            act_space,
            **config["network_kwargs"]     # ✅ pass network config!
        ),
        dataset_class=None,
        **config["alg_kwargs"]
    )

    # 3. Load the trained weights
    checkpoint = torch.load(policy_path, map_location="cpu")
    policy.network.load_state_dict(checkpoint["network"])
    policy.network.eval()

    return policy


def collect_episodes(env, policy, num_episodes, save_path):
    """
    Collects episodes from a trained SAC policy and saves them as .npz files.
    """
    os.makedirs(save_path, exist_ok=True)

    for ep in range(num_episodes):
        obs, info = env.reset()
        episode_obs, episode_actions, episode_rewards = [], [], []
        episode_next_obs, episode_dones = [], []

        done = False
        while not done:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)

            # Use deterministic actor
            with torch.no_grad():
                dist = policy.network.actor(obs_tensor)
                # print(type(dist), dist)
                # action = dist.rsample()
                # action = dist.mean.cpu().numpy()[0]

                action = torch.tanh(dist.base_dist.loc)
                action = action.cpu().numpy()[0]

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # print(reward)
            episode_obs.append(obs)
            episode_actions.append(action)
            episode_rewards.append(reward)
            episode_next_obs.append(next_obs)
            episode_dones.append(done)

            obs = next_obs

        filename = os.path.join(save_path, f"episode_{ep+1:04d}.npz")
        np.savez_compressed(
            filename,
            obs=np.array(episode_obs),
            action=np.array(episode_actions),
            reward=np.array(episode_rewards),
            next_obs=np.array(episode_next_obs),
            done=np.array(episode_dones),
            discount=np.ones(len(episode_dones))
        )

        print(f"[Dataset] Saved episode {ep+1}/{num_episodes} → {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CustomFetchReach-v0")
    parser.add_argument("--policy-path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    env = gym.make(args.env, render_mode="rgb_array" if args.render else None)

    env = FlattenObservation(env) 

    # ✅ Properly rebuild and load policy
    policy = load_trained_policy(env, args.policy_path, args.config)

    # ✅ Collect trajectories
    collect_episodes(env, policy, num_episodes=args.episodes, save_path=args.save_path)
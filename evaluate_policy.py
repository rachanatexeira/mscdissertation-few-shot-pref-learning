import torch
import gymnasium as gym
import numpy as np
from gymnasium.wrappers import FlattenObservation

from research.utils.trainer_gym import load_from_path

def evaluate_policy(policy_path, env_name="CustomFetchReach-v0", episodes=50, render=False):
    # Load environment
    env = gym.make(env_name, render_mode="human" if render else None)
    env = FlattenObservation(env)

    # Load trained policy
    policy = load_from_path(policy_path)
    policy.network.actor.eval()
    device = policy.device

    success_rates = []
    rewards = []

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        successes = 0

        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                dist = policy.network.actor(obs_tensor)
                action = torch.tanh(dist.base_dist.loc).cpu().numpy()[0]

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            successes = info.get("is_success", 0.0)

        rewards.append(total_reward)
        success_rates.append(successes)
        print(f"Episode {ep+1:02d} | Reward = {total_reward:.3f} | Success = {successes:.2f}")

    avg_reward = np.mean(rewards)
    avg_success = np.mean(success_rates)
    print("\n=== Evaluation Results ===")
    print(f"Average Reward: {avg_reward:.3f}")
    print(f"Success Rate:   {avg_success * 100:.2f}%")

if __name__ == "__main__":
    policy_path = "results/fetchreach_run/final_model.pt"  # <-- change if needed
    evaluate_policy(policy_path, env_name="CustomFetchReach-v0", episodes=50, render=False)
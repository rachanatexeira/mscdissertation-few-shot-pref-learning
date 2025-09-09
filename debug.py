import torch
import gymnasium as gym
import gymnasium_robotics
import numpy as np

from research.utils.trainer_gym import load_from_path

from research.envs.custom_fetch_env import CustomFetchReach
from research.networks import ActorCriticPolicy
from research.algs.sac import SAC
from research.datasets import ReplayBuffer
import research.envs  

from gymnasium.wrappers import FlattenObservation

# --- Load environment ---
env = gym.make("CustomFetchReach-v0", render_mode=None)  # <-- change if needed
env = FlattenObservation(env) 
# --- Load trained model ---
model_path = "results/fetchreach_run/final_model.pt"  # <-- change if needed
policy = load_from_path(model_path)
device = policy.device  # automatically set to "mps" if available
policy.network.actor.eval()


# --- Rollout a few episodes ---
for ep in range(10):

    obs, _ = env.reset()
    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    done = False
    step = 0
    print(f"\n=== Episode {ep} ===")
    while not done:
        with torch.no_grad():
            dist = policy.network.actor(obs_tensor)          # returns a SquashedNormal distribution
            # action = dist.mean.cpu().numpy()[0] 



            action = torch.tanh(dist.base_dist.loc)
            action = action.cpu().numpy()[0]

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        print(f"Step {step:02d} | reward={reward:.3f} | distance={info.get('distance', None)} | success={info.get('is_success', None)}")
        step += 1

env.close()
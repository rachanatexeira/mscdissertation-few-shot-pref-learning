import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

from research.networks.mlp import MetaRewardMLPEnsemble 
import gymnasium as gym


class PreferenceDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path, allow_pickle=True)
        self.preferences = data["preferences"]

    def _load_clip(self, file_path):
        traj = np.load(file_path, allow_pickle=True)
        return traj["obs"], traj["action"]

    def __len__(self):
        return len(self.preferences)

    def __getitem__(self, idx):
        item = self.preferences[idx]
        clip1_obs, clip1_act = self._load_clip(item["clip1_file"])
        clip2_obs, clip2_act = self._load_clip(item["clip2_file"])
        label = item["label"]
        r1_env = item["clip1_env_reward"]
        r2_env = item["clip2_env_reward"]

        return (
            torch.tensor(clip1_obs, dtype=torch.float32),
            torch.tensor(clip1_act, dtype=torch.float32),
            torch.tensor(clip2_obs, dtype=torch.float32),
            torch.tensor(clip2_act, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32),
            torch.tensor(r1_env, dtype=torch.float32),
            torch.tensor(r2_env, dtype=torch.float32),
        )

# -----------------------------
# 2. Training Function
# -----------------------------
def train_reward_model(npz_path, save_path, obs_dim, act_dim, epochs=400, batch_size=128, lr=2e-4):
    dataset = PreferenceDataset(npz_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Create gym-style spaces for model init
    observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,))
    action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(act_dim,))

    # Instantiate MetaRewardMLPEnsemble model
    model = MetaRewardMLPEnsemble(
        observation_space=observation_space,
        action_space=action_space,
    )

    optimizer = optim.Adam(model.parameters(), lr=lr)
    # bce = nn.BCEWithLogitsLoss()
    losses = []
    for epoch in range(epochs):
        total_loss = 0
        
        for clip1_obs, clip1_act, clip2_obs, clip2_act, labels, r1_env, r2_env in loader:
            
            b, t, obs_dim = clip1_obs.shape
            _, _, act_dim = clip1_act.shape

            clip1_obs_flat = clip1_obs.view(b * t, obs_dim)
            clip1_act_flat = clip1_act.view(b * t, act_dim)
            clip2_obs_flat = clip2_obs.view(b * t, obs_dim)
            clip2_act_flat = clip2_act.view(b * t, act_dim)

            # print("clip1_obs_flat shape:", clip1_obs_flat.shape)
            # print("model output shape:", model(clip1_obs_flat, clip1_act_flat).shape)
            # print("expected view shape:", (b, t))

            # clip1
            r1_raw = model(clip1_obs_flat, clip1_act_flat)   # (3, B*T)
            r1 = r1_raw.mean(dim=0).view(b, t).mean(dim=1)   # (B,)

            # clip2
            r2_raw = model(clip2_obs_flat, clip2_act_flat)   # (3, B*T)
            r2 = r2_raw.mean(dim=0).view(b, t).mean(dim=1)   # (B,)

            # logits = r1 - r2
            # loss = bce(logits, labels)

            # OPTIONAL: Load env rewards if available in your dataset
            # r1_env = torch.tensor(item["clip1_env_reward"])  # shape: (b,)
            # r2_env = torch.tensor(item["clip2_env_reward"])  # shape: (b,)
            # You could also estimate env reward from obs if you have the reward function

            # alpha = 0.5  # you can tune this
            # r1_env = r1_env.to(r1.device)
            # r2_env = r2_env.to(r2.device)
            # r1_total = alpha * r1 + (1 - alpha) * r1_env
            # r2_total = alpha * r2 + (1 - alpha) * r2_env

            # logits = r1_total - r2_total
            # loss = bce(logits, labels)


            # === Exact Bradley-Terry Loss ===
            # P[clip1 preferred] = exp(r1) / (exp(r1) + exp(r2))
            r1 = r1.to(labels.device)
            r2 = r2.to(labels.device)

            p = torch.exp(r1) / (torch.exp(r1) + torch.exp(r2) + 1e-8)  # avoid zero division
            labels = labels.to(p.device)

            # Compute BCE loss manually using the Bradley-Terry prob
            loss = - (labels * torch.log(p + 1e-8) + (1 - labels) * torch.log(1 - p + 1e-8)).mean()


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f}")
        losses.append(avg_loss)

    os.makedirs(save_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_path, "reward_model.pth"))
    print(f"MetaRewardMLPEnsemble saved to {save_path}/reward_model.pth")

    plt.plot(range(1, epochs + 1), losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Reward Model Training Loss")
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(save_path, "reward_model_training_loss.png")
    plt.savefig(plot_path)

    print(f"📈 Training loss plot saved to {plot_path}")


# -----------------------------
# 3. Run Training
# -----------------------------
if __name__ == "__main__":
    npz_path = "results/fetchreach_run/prefs_dataset/train/simulated_preferences.npz"
    save_path = "results/fetchreach_run/reward_model/"
    obs_dim = 16  # 10 + 3 + 3
    act_dim = 4   # FetchReach has 4D continuous actions
    train_reward_model(npz_path, save_path, obs_dim, act_dim)
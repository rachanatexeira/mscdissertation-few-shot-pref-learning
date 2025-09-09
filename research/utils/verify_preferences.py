import os
import numpy as np
import argparse
import matplotlib.pyplot as plt

def visualize_preference_pair(preference, pair_idx):
    """
    Visualizes and checks one preference pair.
    """
    print(f"\n=== Pair {pair_idx} ===")
    print(f" Clip 1: {os.path.basename(preference['clip1_file'])} | Reward = {preference['clip1_env_reward']:.3f}")
    print(f" Clip 2: {os.path.basename(preference['clip2_file'])} | Reward = {preference['clip2_env_reward']:.3f}")
    print(f" Label: {preference['label']} (1 → prefers clip1, 0 → prefers clip2)")

    # Load observations for both clips
    clip1_data = np.load(preference["clip1_file"])
    clip2_data = np.load(preference["clip2_file"])

    clip1_obs = clip1_data["obs"]
    clip2_obs = clip2_data["obs"]

    # Display reward comparison
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].plot(np.arange(len(clip1_obs)), clip1_data["reward"], color="blue")
    axs[0].set_title(f"Clip 1 Reward Trend ({preference['clip1_env_reward']:.2f})")
    axs[0].set_xlabel("Timestep")
    axs[0].set_ylabel("Reward")

    axs[1].plot(np.arange(len(clip2_obs)), clip2_data["reward"], color="green")
    axs[1].set_title(f"Clip 2 Reward Trend ({preference['clip2_env_reward']:.2f})")
    axs[1].set_xlabel("Timestep")
    axs[1].set_ylabel("Reward")

    plt.suptitle(f"Preference Label = {preference['label']}")
    plt.tight_layout()
    plt.show()


def verify_preferences(pref_path, num_samples=5):
    """
    Randomly samples a few preference pairs and visualizes them.
    """
    data = np.load(pref_path, allow_pickle=True)
    preferences = data["preferences"]

    print(f"✅ Loaded {len(preferences)} preferences from: {pref_path}")
    print(f"🔍 Sampling {num_samples} random preference pairs for inspection...")

    # Pick random sample indices
    sample_indices = np.random.choice(len(preferences), size=min(num_samples, len(preferences)), replace=False)

    for idx in sample_indices:
        visualize_preference_pair(preferences[idx], idx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pref-path", type=str, required=True, help="Path to simulated_preferences.npz")
    parser.add_argument("--samples", type=int, default=5, help="Number of preference pairs to visualize")
    args = parser.parse_args()

    verify_preferences(args.pref_path, args.samples)
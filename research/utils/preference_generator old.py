import os
import numpy as np
import random
import argparse

def load_clip(file_path, clip_len=25):
    """Load a random clip from a trajectory and calculate its cumulative reward."""
    data = np.load(file_path)
    rewards = data["rewards"]
    obs = data["observations"]

    # Pick a random starting point for the clip
    start = random.randint(0, max(0, len(obs) - clip_len))
    end = start + clip_len

    # Slice observations + sum reward over the clip
    clip_obs = obs[start:end]
    clip_reward = rewards[start:end].sum()

    return clip_obs, clip_reward

def generate_preferences(traj_dir, save_path, num_pairs=1000, clip_len=25):
    """Generate simulated preferences from stored trajectories."""
    os.makedirs(save_path, exist_ok=True)
    traj_files = [os.path.join(traj_dir, f) for f in os.listdir(traj_dir) if f.endswith(".npz")]

    assert len(traj_files) >= 2, "Need at least two trajectories to generate preferences!"

    prefs = []
    for i in range(num_pairs):
        # Sample two random episodes
        f1, f2 = random.sample(traj_files, 2)

        # Load random clips and compute rewards
        clip1, r1 = load_clip(f1, clip_len)
        clip2, r2 = load_clip(f2, clip_len)

        # Preference label: 1 if clip1 better, else 0
        label = 1 if r1 > r2 else 0

        prefs.append({
            "clip1_file": f1,
            "clip2_file": f2,
            "clip1_reward": r1,
            "clip2_reward": r2,
            "label": label
        })

        if (i + 1) % 100 == 0:
            print(f"[{i + 1}/{num_pairs}] preferences generated.")

    # Save preferences to disk
    np.savez_compressed(
        os.path.join(save_path, "simulated_preferences.npz"),
        preferences=prefs
    )

    print(f"\n✅ Generated {num_pairs} simulated preferences → {save_path}/simulated_preferences.npz")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj-dir", type=str, required=True, help="Path to directory containing .npz trajectories")
    parser.add_argument("--save-path", type=str, required=True, help="Where to save simulated preferences")
    parser.add_argument("--num-pairs", type=int, default=1000, help="Number of preference pairs to generate")
    parser.add_argument("--clip-len", type=int, default=25, help="Length of each clip in timesteps")
    args = parser.parse_args()

    generate_preferences(args.traj_dir, args.save_path, args.num_pairs, args.clip_len)
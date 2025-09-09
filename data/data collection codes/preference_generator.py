import os
import numpy as np
import random
import argparse

def load_clip(file_path, clip_len=25):
    """Load a random clip from a trajectory and return observations and actions."""
    data = np.load(file_path)
    # print(data)
    obs = data["obs"]
    act = data["action"]
    rew = data["reward"]
    dis = data["discount"]
    done = data["done"]

    # Random start
    start = random.randint(0, max(0, len(obs) - clip_len))
    end = start + clip_len

    clip_obs = obs[start:end]
    clip_act = act[start:end]
    clip_rew = rew[start:end]
    clip_dis = dis[start:end]
    clip_done = done[start:end]

    return clip_obs, clip_act, clip_rew, clip_dis, clip_done

def generate_preferences(traj_dir, save_path, num_pairs=1000, clip_len=25):
    """Generate simulated preferences by saving clip files and referencing them in the dataset."""
    os.makedirs(os.path.join(save_path, "clips"), exist_ok=True)
    traj_files = [os.path.join(traj_dir, f) for f in os.listdir(traj_dir) if f.endswith(".npz")]
    assert len(traj_files) >= 2, "Need at least two trajectories!"

    prefs = []
    for i in range(num_pairs):
        f1, f2 = random.sample(traj_files, 2)
        obs1, act1, rew1, dis1, do1 = load_clip(f1, clip_len)
        obs2, act2, rew2, dis2, do2 = load_clip(f2, clip_len)

        # Save clip1
        clip1_path = os.path.join(save_path, "clips", f"clip_{i}_1.npz")
        np.savez_compressed(clip1_path, obs=obs1, action=act1, reward=rew1, discount=dis1, done=do1)

        # Save clip2
        clip2_path = os.path.join(save_path, "clips", f"clip_{i}_2.npz")
        np.savez_compressed(clip2_path, obs=obs2, action=act2, reward=rew2, discount=dis2, done=do2)

        # # Preference label: 1 if clip1 is better
        # label = float(rew1.sum() > rew2.sum())
        p = np.exp(rew1.sum()) / (np.exp(rew1.sum()) + np.exp(rew2.sum()))

        # label = int(np.random.rand() < p)
        label = 1 if int(np.random.rand() < p) else 2

        prefs.append({
            "clip1_file": clip1_path,
            "clip2_file": clip2_path,
            "clip1_env_reward": float(rew1.sum()),
            "clip2_env_reward": float(rew2.sum()),
            "label": label,
        })

        if (i + 1) % 100 == 0:
            print(f"[{i + 1}/{num_pairs}] preferences generated")

    # Save preferences
    np.savez_compressed(os.path.join(save_path, "simulated_preferences.npz"), preferences=prefs)
    print(f"\n✅ Saved preference dataset → {os.path.join(save_path, 'simulated_preferences.npz')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj-dir", type=str, required=True, help="Path to trajectory .npz files")
    parser.add_argument("--save-path", type=str, required=True, help="Where to save preferences + clips")
    parser.add_argument("--num-pairs", type=int, default=1000)
    parser.add_argument("--clip-len", type=int, default=25)
    args = parser.parse_args()

    generate_preferences(args.traj_dir, args.save_path, args.num_pairs, args.clip_len)
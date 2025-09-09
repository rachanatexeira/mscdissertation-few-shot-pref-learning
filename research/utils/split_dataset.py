import os
import numpy as np
import shutil
import argparse
from sklearn.model_selection import train_test_split

def split_dataset(pref_path, output_dir, split_ratio=0.8):
    data = np.load(pref_path, allow_pickle=True)
    preferences = data["preferences"]

    # Split preferences
    train_prefs, valid_prefs = train_test_split(preferences, train_size=split_ratio, random_state=42)

    # Copy clips to train/ and valid/
    for split_name, split_prefs in [("train", train_prefs), ("valid", valid_prefs)]:
        split_dir = os.path.join(output_dir, split_name)
        clips_dir = os.path.join(split_dir, "clips")
        os.makedirs(clips_dir, exist_ok=True)

        for item in split_prefs:
            for clip_key in ["clip1_file", "clip2_file"]:
                src = item[clip_key]
                dst = os.path.join(clips_dir, os.path.basename(src))
                if not os.path.exists(dst):
                    shutil.copy(src, dst)
                item[clip_key] = dst  # Update path

        # Save split preferences
        np.savez_compressed(os.path.join(split_dir, "simulated_preferences.npz"), preferences=split_prefs)
        print(f"✅ Saved {len(split_prefs)} to {split_name}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pref-path", required=True, help="Path to original simulated_preferences.npz")
    parser.add_argument("--output-dir", required=True, help="Path to save split train/ and valid/")
    parser.add_argument("--split-ratio", type=float, default=0.8, help="Train split ratio (default=0.8)")
    args = parser.parse_args()

    split_dataset(args.pref_path, args.output_dir, args.split_ratio)
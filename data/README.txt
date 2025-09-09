Few-Shot Preference Learning – Supplementary Material
-----------------------------------------------------

This folder contains the primary data, software code, and logs used in the dissertation titled:

"Few-Shot Preference Learning for Human-in-the-Loop Reinforcement Learning"

Folder structure:


1. configs/
   - sac.yaml - hyperparameter configs for training SAC
   - maml.yaml - hyperparameter configs for maml

2. trajectory_data/
   - *.npz: Recorded trajectories from a pre-trained SAC agent.

3. preference_data/
   - all/clips/clip_*_*.npz - preferenc pairs
   - all/simulated_preferences.npz
   - train/clips - training set
   - valid/clips - validation set
4. models/
   - sac/best_model.pt: baseline SAC model
   - maml/best_model.pt: maml trained model
   - reward model/reward_model.pth - reward model generated

5. data collection codes/
   - collect_policy_dataset.py: Collects trajectories using SAC agent.
   - preference_generator.py: Creates preference labels from trajectory segments.
   - train_reward_model.py: Trains the reward network using Preference-MAML.

6. logs/
   - reward_model_log.txt: Logs of reward model training (loss, accuracy).
   - sac_training_log.txt: Logs of SAC training (returns, steps).

Contact: Rachana Texeira | MSc AI Dissertation, Sheffield Hallam University
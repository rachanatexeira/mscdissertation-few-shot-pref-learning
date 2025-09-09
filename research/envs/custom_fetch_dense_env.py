import numpy as np
import gymnasium as gym
from gymnasium_robotics.envs.fetch.pick_and_place import FetchPickAndPlaceEnv  
import mujoco

class CustomFetchPickAndPlaceDenseEnv(FetchPickAndPlaceDenseEnv):
    def __init__(self, **kwargs):
        kwargs["reward_type"] = "dense"
        super().__init__(**kwargs)

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)

        # === Get cube joint index ===
        object_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "object0:joint")
        object_qpos_addr = self.model.jnt_qposadr[object_joint_id]
        qpos = self.data.qpos.copy()

        # === Recreate original sampling: keep retrying until cube is 0.1m away ===
        gripper_xy = qpos[:2]  # gripper x, y
        cube_xy = np.zeros(2)
        attempts = 0

        while True:
            cube_xy = gripper_xy + np.random.uniform(-0.15, 0.15, size=2)
            if np.linalg.norm(cube_xy - gripper_xy) >= 0.1:
                break
            attempts += 1
            if attempts > 100:  # fallback if stuck
                cube_xy = gripper_xy + np.array([0.1, 0.0])
                break

        # === Set cube x, y ===
        qpos[object_qpos_addr + 0] = cube_xy[0]  # x
        qpos[object_qpos_addr + 1] = cube_xy[1]  # y
        qpos[object_qpos_addr + 2] = 0.425       # ✅ fixed z on table

        self.data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.data)

        obs = self._get_obs()

        print(f"[CustomReset] Cube pos: {cube_xy[0]:.3f}, {cube_xy[1]:.3f}, z=0.425")
        return obs, info
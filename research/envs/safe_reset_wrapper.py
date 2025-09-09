from gymnasium import Wrapper
import numpy as np
import mujoco

class SafeResetWrapper(Wrapper):
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        print("obs before:", obs["observation"][:6])

        # Recursively unwrap base env
        def get_base_env(env):
            while hasattr(env, "env"):
                env = env.env
            return env

        env = get_base_env(self.env)

        # Set object height if needed (optional)
        object_joint_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "object0:joint")
        object_qpos_addr = env.model.jnt_qposadr[object_joint_id]

        # [Optional] Set cube height only
        qpos = env.data.qpos.copy()
        qpos[object_qpos_addr + 2] = 0.425  # Set Z only (keep X/Y as is)

        # ✅ Move the gripper
        grip_joint_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, "robot0:grip")
        # Estimate gripper position index (usually from qpos[0:3])
        # Set gripper X/Y closer to cube
        qpos[0] = 0.0  # X
        qpos[1] = 0.5  # Y
        # Note: Do NOT change Z or orientation for now

        env.data.qpos[:] = qpos
        mujoco.mj_forward(env.model, env.data)

        obs = env._get_obs()
        print("obs after:", obs["observation"][:6])
        # print(f"[SafeReset] Gripper moved to x={qpos[0]:.3f}, y={qpos[1]:.3f}")
        return obs, info




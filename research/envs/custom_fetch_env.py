import numpy as np
from gymnasium_robotics.envs.fetch.reach import MujocoFetchReachEnv
from gymnasium.utils.ezpickle import EzPickle

class CustomFetchReach(MujocoFetchReachEnv):
    def __init__(self, **kwargs):
        # print("⚡ DEBUG: CustomFetchReach init called with kwargs:", kwargs)
        super().__init__(**kwargs)
        EzPickle.__init__(self, **kwargs)
        # print(getattr(self, "reward_type", "NOT FOUND"))
        self.reward_type = "dense"  # Force override


    # def _reset_sim(self):
    #     print("💡 [DEBUG] Called: CustomFetchPickAndPlaceEnv.reset - 1")
    #     # Reset buffers for joint states, actuators, warm-start, control buffers etc.
    #     self._mujoco.mj_resetData(self.model, self.data)

    #     # Randomize start position of object.
    #     if self.has_object:
    #         object_xpos = self.initial_gripper_xpos[:2]

            
    #         # object_xpos
    #         while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1:
    #             # object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(
    #             #     -self.obj_range, self.obj_range, size=2
    #             # )
    #             x_offset = self.np_random.uniform(-0.55, -0.15)   # always in front of gripper
    #             y_offset = self.np_random.uniform(-0.75, -0.15)  # very narrow band around center
    #             object_xpos = self.initial_gripper_xpos[:2] + np.array([x_offset, y_offset])


    #         object_qpos = self._utils.get_joint_qpos(
    #             self.model, self.data, "object0:joint"
    #         )
    #         assert object_qpos.shape == (7,)
    #         object_qpos[:2] = object_xpos
    #         self._utils.set_joint_qpos(
    #             self.model, self.data, "object0:joint", object_qpos
    #         )

    #     self._mujoco.mj_forward(self.model, self.data)
    #     return True



    def _sample_goal(self):
        x_offset = self.np_random.uniform(-0.45, -0.15)   # always in front of gripper
        y_offset = self.np_random.uniform(-0.75, -0.15)  # very narrow band around center
        z_offset = self.np_random.uniform(-0.15, 0.15)  # very narrow band around center
        if self.has_object:
            # goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(
            #     -self.target_range, self.target_range, size=3
            # )
            goal = self.initial_gripper_xpos[:3] + np.array([x_offset, y_offset, z_offset])
            goal += self.target_offset
            goal[2] = self.height_offset
            if self.target_in_the_air and self.np_random.uniform() < 0.5:
                goal[2] += self.np_random.uniform(0, 0.5)
        else:
            # goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(
            #     -self.target_range, self.target_range, size=3
            # )
            goal = self.initial_gripper_xpos[:3] + np.array([x_offset, y_offset, z_offset])
        return goal.copy()


    def compute_reward(self, achieved_goal, desired_goal, info):
        if self.reward_type == "dense":
            # Force dense reward manually
            return -np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        else:
            # Fallback to parent sparse reward
            return super().compute_reward(achieved_goal, desired_goal, info)
        


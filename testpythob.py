import gymnasium as gym

envs = gym.envs.registry
for env in sorted(envs.keys()):
    print(env)
# main file for the project

import numpy as np
import matplotlib.pyplot as plt
from visualization import plot_map, plot_map_with_path, plot_map_with_tree_path
from rrt_env import RRTEnv

def test_visualization():
    # obstacle positions and dimensions
    obstacles = [
        (0, 5, 5, 2),
        (3, 5, 2, 10)

    ]
    start = (0,0)
    goal = (10, 10)

    # plot map
    plot_map(obstacles, start, goal)

    path = [(1, 1), (3, 3), (10, 10)]
    plot_map_with_path(obstacles, start, goal, path)




# test RRT on toy example
from rrt import RRT

def test_rrt():
    # obstacle positions and dimensions
    obstacles = [
        (0, 2, 4, 2),
        (3, 5, 5, 5),
        (8, 8, 1.9, 2)

    ]
    start = (0, 0)
    goal = (10, 10)

    rrt = RRT(start, goal, obstacles, (10, 10), 2000, 0.5, 0.09)
    tree, total_collision_checks = rrt.plan()
    print("number of nodes in generated tree:", len(rrt.nodes))
    print("total collision checks:", total_collision_checks)
    plot_map_with_tree_path(obstacles, start, goal, tree)





def test_rrt_env():

    # test the environment
    env = RRTEnv((10, 10), 3, 2000, 0.5, 0.09)
    env.reset()
    obstacles = np.array([0, 5, 5, 2, 3, 5, 2, 10, 8, 8, 1.9, 2])
    obs, reward = env.step(obstacles)
    print("reward:", reward)

    if(env.getTree() is None):
        plot_map(obstacles, obs[0], obs[1])
    else:
        plot_map_with_tree_path(obstacles, obs[0], obs[1], env.getTree())


# test_rrt()
# test_visualization()
# test_rrt_env()


# run TD3 on the environment
import gym

from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise


from gymnasium.envs.registration import register

register(
     id="my_env-v0",
     entry_point="rrt_env:RRTEnv",
     max_episode_steps=300,
)






#env = gym.make("my_env-v0")
env = RRTEnv(map_size=(1, 1), num_obstacles=2, max_iter=200, step_size=0.1, reachability_step_size=0.04)

# The noise objects for TD3
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=1)
model.learn(total_timesteps=2000, log_interval=10)

# model.save("td3_pendulum")
# vec_env = model.get_env()
# 
# del model # remove to demonstrate saving and loading
# 
# model = TD3.load("td3_pendulum")

# eval loop
while True:
    obs, _ = env.reset()
    action, _states = model.predict(obs)
    obs, rewards, terminated, truncated, info = env.step(action)
    env.render("human")
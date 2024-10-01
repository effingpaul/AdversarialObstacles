# create an OPEN GYM API environment which creates single step RRT paths

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
from rrt import RRT
from visualization import plot_map, plot_map_with_path, plot_map_with_tree_path

class RRTEnv(gym.Env):
    def __init__(self, map_size, num_obstacles, max_iter, step_size, reachability_step_size):
        self.map_size = map_size
        self.num_obstacles = num_obstacles
        self.max_iter = max_iter
        self.step_size = step_size
        self.surface_weight = 100
        self.tree = None
        self.reachability_step_size = reachability_step_size
        # actions are the positions of the obstacles (5 coordinates for each obstacle)
        self.action_space = spaces.Box(low=0, high=map_size[0], shape=(5 * num_obstacles,), dtype=np.float32)
        # observations are the start and goal positions
        self.observation_space = spaces.Box(low=0, high=map_size[0], shape=(4,), dtype=np.float32)
        self.seed()
        self.reset()

    def _get_info(self):
        return {
        "placeholder": 0
        }

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def reset(self, seed=None):
        self.start = (random.uniform(0, self.map_size[0]), random.uniform(0, self.map_size[1]))
        self.goal = (random.uniform(0, self.map_size[0]), random.uniform(0, self.map_size[1]))
        # redo if distance too small
        while np.linalg.norm(np.array(self.start) - np.array(self.goal)) < 0.2 * np.linalg.norm(np.array(self.map_size)):
            self.goal = (random.uniform(0, self.map_size[0]), random.uniform(0, self.map_size[1]))
        self.obstacles = []
        self.tree= []
        # observations are flattened start and goal poitions so total length of 5
        observation = np.array(self.start + self.goal)
        return observation, "no info"

    def step(self, action):
        # set the obstacles
        self.obstacles = []
        # observations are flattened start and goal poitions so total length of 5
        observation = np.array(self.start + self.goal)
        for i in range(self.num_obstacles):
            self.obstacles.append(action[i*5:i*5+5])
            self.obstacles[-1][4] += 0.1 # add minimum width to obstacles # TODO: place this somewhere else
        
        # run the RRT algorithm
        rrt = RRT(self.start, self.goal, self.obstacles, self.map_size, self.max_iter, self.step_size, self.reachability_step_size)
        if not rrt.check_goal_and_start_are_not_occupied():
            print("start or goal was occupied!")
            reward = -self.surface_weight * rrt.get_obstacle_surface(surface_base_cost=0.1) #surface base cost is what one obstacle will at least always cost
            return observation, reward, True, False, self._get_info()
        # the reward is the number of nodes of the tree
        tree, total_collision_checks = rrt.plan()
        self.tree = tree
        reward = len(tree) - self.surface_weight * rrt.get_obstacle_surface()
        
        return observation, reward, True, False, self._get_info()
    
    def render(self, mode='human'):
        if mode == 'human':
            print(self.obstacles)
            plot_map_with_tree_path(self.obstacles, self.start, self.goal, self.tree)
        

    def getTree(self):
        return self.tree


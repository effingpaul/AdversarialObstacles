# This implements standard RRT algorithm for 2D path planning

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random

class RRT:
    def __init__(self, start, goal, obstacles, map_size, max_iter, step_size, reachability_step_size):
        self.start = start
        self.goal = goal
        self.obstacles = obstacles
        self.map_size = map_size
        self.max_iter = max_iter
        self.step_size = step_size
        self.reachability_step_size = reachability_step_size
        self.collision_checks = 0
        self.nodes = [start]

    def is_occupied(self, point):
        # an obstacle is a tupel of (x1, y1, x2, y2, width)
        # it is a rectangle defined by the line between (x1, y1) and (x2, y2) and the width
        self.collision_checks += 1
        for obs in self.obstacles:
            if obs[4] == 0:
                continue
            #if (obs[0] - self.reachability_step_size <= point[0] <= obs[0] + obs[2] + self.reachability_step_size
            #   and obs[1]- self.reachability_step_size  <= point[1] <= obs[1] + obs[3] + self.reachability_step_size):
            #    return True
            # for collision we have to check if the ppoint is beyond the 4 lines that define the rectangle
            # first find the orthoginal vector of the line. it is the vetor that stands in a 90 degree angle to the line
            line = np.array([obs[2] - obs[0], obs[3] - obs[1]])
            orth = np.array([line[1], -line[0]])
            orth = orth / np.linalg.norm(orth)
            # now we have to find the 4 points that define the rectangle
            p2 = np.array([obs[0], obs[1]]) + orth * obs[4] / 2
            p1 = np.array([obs[0], obs[1]]) - orth * obs[4] / 2
            p3 = np.array([obs[2], obs[3]]) + orth * obs[4] / 2
            p4 = np.array([obs[2], obs[3]]) - orth * obs[4] / 2
            # now we have to check if the point is on the right side of the 4 lines
            # if the point is on the right side of all 4 lines it is inside the rectangle
            # do it liek this: #
            # 0 <= dot(AB,AM) <= dot(AB,AB) &&
            # 0 <= dot(BC,BM) <= dot(BC,BC)
            p1_p2 = p2-p1
            p1_point = point - p1
            p1_p2_dot_p1_point = np.dot(p1_p2, p1_point)
            p1_p2_dot_p1_p2 = np.dot(p1_p2, p1_p2)
            if 0 <= p1_p2_dot_p1_point <= p1_p2_dot_p1_p2:
                p2_p3 = p3-p2
                p2_point = point - p2
                p2_p3_dot_p2_point = np.dot(p2_p3, p2_point)
                p2_p3_dot_p2_p3 = np.dot(p2_p3, p2_p3)
                if 0 <= p2_p3_dot_p2_point <= p2_p3_dot_p2_p3:
                    return True
        return False
    
    def get_obstacle_surface(self, surface_base_cost=1):
        surface = 0
        for obs in self.obstacles:
            # the surface of a rectangle is width * height
            # width is given and hieght is the distance between the two points
            surface_obs = obs[4] * np.linalg.norm(np.array([obs[0], obs[1]]) - np.array([obs[2], obs[3]]))
            surface += min(surface_obs, surface_base_cost)
            # surface += obs[4] * np.linalg.norm(np.array([obs[0], obs[1]]) - np.array([obs[2], obs[3]]))

        
        return surface
    
    def check_goal_and_start_are_not_occupied(self):
        return not self.is_occupied(self.start) and not self.is_occupied(self.goal)
    
    def sample_point(self):
        while True:
            point = (random.uniform(0, self.map_size[0]), random.uniform(0, self.map_size[1]))
            if not self.is_occupied(point):
                return point
    
    def nearest_node(self, point):
        distances = [np.linalg.norm(np.array(node) - np.array(point)) for node in self.nodes]
        return self.nodes[np.argmin(distances)]
    
    def new_point(self, nearest, point):
        diff = np.array(point) - np.array(nearest)
        diff_norm = np.linalg.norm(diff)
        if diff_norm < self.step_size:
            return point
        return nearest + self.step_size * diff / diff_norm
    
    def is_reachable(self, nearest, new, reachability_step_size=0.1):
        diff = np.array(new) - np.array(nearest)
        diff_norm = np.linalg.norm(diff)
        num_steps = int(diff_norm / reachability_step_size)
        for i in range(num_steps):
            point = nearest + i * reachability_step_size * diff / diff_norm
            if self.is_occupied(point):
                return False
        return True
    
    def plan(self):
        tree = []
        for i in range(self.max_iter):
            point = self.sample_point()
            nearest = self.nearest_node(point)
            new = self.new_point(nearest, point)
            if self.is_reachable(nearest, new, self.reachability_step_size):
                self.nodes.append(new)
                tree.append((nearest, new))
                if self.is_reachable(new, self.goal, self.reachability_step_size):
                    tree.append((new, self.goal))
                    return tree, self.collision_checks
        return tree, self.collision_checks
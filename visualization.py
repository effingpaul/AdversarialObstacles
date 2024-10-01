# defines the visualization for a 2D map based on coordinates of the rectangle obstacles

import matplotlib.pyplot as plt
import numpy as np

def plot_map(obstacles, start, goal):
    """
    Plots the 2D map with obstacles and start and goal positions
    :param obstacles: list of lists of coordinates of the rectangle obstacles
    :param start: list of coordinates of the start position
    :param goal: list of coordinates of the goal position
    :return: None
    """
    fig, ax = plt.subplots()

    for obs in obstacles:
        # obstacles are defined by two points and a width
        line = np.array([obs[2] - obs[0], obs[3] - obs[1]])
        orth = np.array([line[1], -line[0]])
        orth = orth / np.linalg.norm(orth)
        p1 = np.array([obs[0], obs[1]]) + orth * obs[4] / 2
        p2 = np.array([obs[0], obs[1]]) - orth * obs[4] / 2
        p3 = np.array([obs[2], obs[3]]) + orth * obs[4] / 2
        p4 = np.array([obs[2], obs[3]]) - orth * obs[4] / 2
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'black')
        ax.plot([p2[0], p3[0]], [p2[1], p3[1]], 'black')
        ax.plot([p3[0], p4[0]], [p3[1], p4[1]], 'black')
        ax.plot([p4[0], p1[0]], [p4[1], p1[1]], 'black')

    ax.plot(start[0], start[1], 'bo', markersize=10)
    ax.plot(goal[0], goal[1], 'ro', markersize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


def plot_map_with_path(obstacles, start, goal, path): 
    """
    Plots the 2D map with obstacles, start and goal positions, and the path
    :param obstacles: list of lists of coordinates of the rectangle obstacles
    :param start: list of coordinates of the start position
    :param goal: list of coordinates of the goal position
    :param path: list of coordinates of the path
    :return: None
    """
    fig, ax = plt.subplots()

    for obs in obstacles:
        # obstacles are defined by two points and a width
        line = np.array([obs[2] - obs[0], obs[3] - obs[1]])
        orth = np.array([line[1], -line[0]])
        orth = orth / np.linalg.norm(orth)
        p1 = np.array([obs[0], obs[1]]) + orth * obs[4] / 2
        p2 = np.array([obs[0], obs[1]]) - orth * obs[4] / 2
        p3 = np.array([obs[2], obs[3]]) + orth * obs[4] / 2
        p4 = np.array([obs[2], obs[3]]) - orth * obs[4] / 2
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'black')
        ax.plot([p2[0], p3[0]], [p2[1], p3[1]], 'black')
        ax.plot([p3[0], p4[0]], [p3[1], p4[1]], 'black')
        ax.plot([p4[0], p1[0]], [p4[1], p1[1]], 'black')

    ax.plot(start[0], start[1], 'bo', markersize=10)
    ax.plot(goal[0], goal[1], 'ro', markersize=10)
    for i in range(len(path)-1):
        ax.plot([path[i][0], path[i+1][0]], [path[i][1], path[i+1][1]], 'g')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

def plot_map_with_tree_path(obstacles, start, goal, tree):
    """
    Plots the 2D map with obstacles, start and goal positions, and the tree path
    :param obstacles: list of lists of coordinates of the rectangle obstacles
    :param start: list of coordinates of the start position
    :param goal: list of coordinates of the goal position
    :param tree: list of tuples of two connected tree notes (a note are two coordinates)
    :return: None
    """
    fig, ax = plt.subplots()

    for obs in obstacles:
        # obstacles are defined by two points and a width
        line = np.array([obs[2] - obs[0], obs[3] - obs[1]])
        orth = np.array([line[1], -line[0]])
        orth = orth / np.linalg.norm(orth)
        p2 = np.array([obs[0], obs[1]]) + orth * obs[4] / 2
        p3 = np.array([obs[0], obs[1]]) - orth * obs[4] / 2
        p1 = np.array([obs[2], obs[3]]) + orth * obs[4] / 2
        p4 = np.array([obs[2], obs[3]]) - orth * obs[4] / 2
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'black')
        ax.plot([p2[0], p3[0]], [p2[1], p3[1]], 'black')
        ax.plot([p3[0], p4[0]], [p3[1], p4[1]], 'black')
        ax.plot([p4[0], p1[0]], [p4[1], p1[1]], 'black')

    ax.plot(start[0], start[1], 'bo', markersize=10)
    ax.plot(goal[0], goal[1], 'ro', markersize=10)
    for edge in tree:
        ax.plot([edge[0][0], edge[1][0]], [edge[0][1], edge[1][1]], 'b')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
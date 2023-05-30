##############################################################################

# import packages

##############################################################################

from PIL import Image
from random import randint as rand
import numpy as np
from collections import deque
import heapq

import matplotlib.pyplot as plt

from matplotlib.pyplot import figure

##############################################################################

# plot grid

##############################################################################

img = Image.open('randomhardcore_map.png')
# Convert the image to grayscale
gray_img = img.convert('L')

# Convert the grayscale image to a numpy array
img_array = np.array(gray_img)

# Threshold the image to get black and white
threshold = 128
grid = np.where(img_array < threshold, 1, 0)

def valid_point(point, grid):
    return 0 <= point[0] < grid.shape[0] and 0 <= point[1] < grid.shape[1] and grid[point] == 0

# start point and goal
start, goal = (rand(1,85), rand(1,85)), (rand(1,85), rand(1,85))

# If the points aren't valid, regenerate them
while not valid_point(start, grid) or not valid_point(goal, grid):
    start, goal = (rand(1,85), rand(1,85)), (rand(1,85), rand(1,85))


##############################################################################

# heuristic function for path scoring

##############################################################################


def heuristic(a, b, astar=True):
    if astar:
        return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)
    else:
        return 0


##############################################################################

# path finding function

##############################################################################


def path_finder(array, start, goal, astar=True, diagonal=True):
    if diagonal:
        neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    else:
        neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    close_set = set()
    came_from = {}
    gscore = {start: 0}
    fscore = {start: heuristic(start, goal, astar=astar)}
    oheap = []
    heapq.heappush(oheap, (fscore[start], start))
    while oheap:
        current = heapq.heappop(oheap)[1]
        if current == goal:
            data = []
            while current in came_from:
                data.append(current)
                current = came_from[current]
            return data, close_set
        close_set.add(current)
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j
            tentative_g_score = gscore[current] + heuristic(current, neighbor, astar=True)
            if 0 <= neighbor[0] < array.shape[0]:
                if 0 <= neighbor[1] < array.shape[1]:
                    if array[neighbor[0]][neighbor[1]] == 1:
                        continue
                else:
                    # array bound y walls
                    continue
            else:
                # array bound x walls
                continue

            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                continue

            if tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1] for i in oheap]:
                came_from[neighbor] = current

                gscore[neighbor] = tentative_g_score

                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal, astar=astar)

                heapq.heappush(oheap, (fscore[neighbor], neighbor))

    return False

# For BFS and DFS
def reconstruct_path(came_from, start, goal):
    current = goal
    path = []
    while current != start:
        path.append(current)
        current = came_from[current]
    path.append(start)  # optional
    path.reverse()  # optional
    return path

def get_neighbors(map, node):
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    neighbors = []
    for dx, dy in directions:
        nx, ny = node[0] + dx, node[1] + dy
        if nx >= 0 and nx < len(map) and ny >= 0 and ny < len(map[0]) and map[nx][ny] == 0:
            neighbors.append((nx, ny))
    return neighbors

def dfs(map, start, goal):
    stack = [start]
    came_from = {start: None}
    visited = []

    while stack:
        node = stack.pop()
        visited.append(node)

        if node == goal:
            return reconstruct_path(came_from, start, goal), visited

        for neighbor in get_neighbors(map, node):
            if neighbor not in came_from:
                stack.append(neighbor)
                came_from[neighbor] = node

    return None, visited

def bfs(map, start, goal):
    queue = deque([start])
    came_from = {start: None}
    visited = []

    while queue:
        node = queue.popleft()
        visited.append(node)

        if node == goal:
            return reconstruct_path(came_from, start, goal), visited

        for neighbor in get_neighbors(map, node):
            if neighbor not in came_from:
                queue.append(neighbor)
                came_from[neighbor] = node
    return None, visited

bfs_route, bfs_visited = bfs(grid, start, goal)
dfs_route, dfs_visited = dfs(grid, start, goal)
d_route, d_closed_set = path_finder(grid, start, goal, astar=False)
a_route, a_closed_set = path_finder(grid, start, goal, astar=True)

#dfs_route = dfs_route + [start]
#bfs_route = bfs_route + [start]
d_route = d_route + [start]
a_route = a_route + [start]

dfs_route = dfs_route[::-1]
bfs_route = bfs_route[::-1]
d_route = d_route[::-1]
a_route = a_route[::-1]
if len(a_route) == 0:
    exit('no route')
##############################################################################

# plot the path

##############################################################################
print("BFS Searched points: ", len(bfs_visited))
print("BFS Path length: ", len(bfs_route))

print("DFS Searched points: ", len(dfs_visited))
print("DFS Path length: ", len(dfs_route))

print("Djikstras Searched points: ", len(d_closed_set))
print("Djikstras Path length: ", len(d_route))

print("A* Searched points: ", len(a_closed_set))
print("A* Path length: ", len(a_route))

# Convert the sets into numpy arrays
bfs_path_points = np.array(list(bfs_route))
bfs_searched_points = np.array(list(bfs_visited))

dfs_path_points = np.array(list(dfs_route))
dfs_searched_points = np.array(list(dfs_visited))

d_path_points = np.array(list(d_route))
d_searched_points = np.array(list(d_closed_set))

a_path_points = np.array(list(a_route))
a_searched_points = np.array(list(a_closed_set))

# plot map and path
fig, ax = plt.subplots(4, figsize=(20, 20))

ax[0].imshow(grid, cmap=plt.cm.Dark2)
ax[0].set_title('Dijkstras', fontsize=20)
ax[1].imshow(grid, cmap=plt.cm.Dark2)
ax[1].set_title('A*', fontsize=20)
ax[2].imshow(grid, cmap=plt.cm.Dark2)
ax[2].set_title('BFS', fontsize=20)
ax[3].imshow(grid, cmap=plt.cm.Dark2)
ax[3].set_title('DFS', fontsize=20)

ax[0].scatter(start[1], start[0], marker="*", color="yellow", s=100)
ax[0].scatter(goal[1], goal[0], marker="*", color="red", s=100)
ax[1].scatter(start[1], start[0], marker="*", color="yellow", s=100)
ax[1].scatter(goal[1], goal[0], marker="*", color="red", s=100)
ax[2].scatter(start[1], start[0], marker="*", color="yellow", s=100)
ax[2].scatter(goal[1], goal[0], marker="*", color="red", s=100)
ax[3].scatter(start[1], start[0], marker="*", color="yellow", s=100)
ax[3].scatter(goal[1], goal[0], marker="*", color="red", s=100)

ax[0].scatter(d_searched_points[:,1], d_searched_points[:,0], marker='o', color="blue", s=1)
ax[1].scatter(a_searched_points[:,1], a_searched_points[:,0], marker='o', color="blue", s=1)
ax[2].scatter(bfs_searched_points[:,1], bfs_searched_points[:,0], marker='o', color="blue", s=1)
ax[3].scatter(dfs_searched_points[:,1], dfs_searched_points[:,0], marker='o', color="blue", s=1)

ax[0].plot(d_path_points[:,1], d_path_points[:,0], color="black", linewidth=2)
ax[1].plot(a_path_points[:,1], a_path_points[:,0], color="black", linewidth=2)
ax[2].plot(bfs_path_points[:,1], bfs_path_points[:,0], color="black", linewidth=2)
ax[3].plot(dfs_path_points[:,1], dfs_path_points[:,0], color="black", linewidth=2)

plt.show()
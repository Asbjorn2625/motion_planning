import shapely.geometry as sg
from shapely import affinity
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.spatial import ConvexHull
from scipy.stats import qmc
import heapq


# Check if a point is visible from another point
def is_visible(p1, p2, obstacles_list):
    line = sg.LineString([p1, p2])
    for obstacle in obstacles_list:
        if line.intersects(obstacle) and not line.touches(obstacle):
            return False
    return True

# Class for the proberlistic map
class ProberlisticMap:
    def __init__(self, size, obstacles, num_points=100, max_checks=None):
        self.obstacles = obstacles
        self.size = size
        self.points = self.generate_random_points(num_points)
        self.visibility_graph = self.generate_visibility_graph(max_checks)

    # Generate random points using halton sequence
    def generate_random_points(self, num_points):
        hamton = qmc.Halton(2)
        samples = hamton.random(num_points)
        return [sg.Point(sample[0] * self.size, sample[1] * self.size) for sample in samples]

    # Add new point to the visibility graph
    def add_point_to_visibility_graph(self, point):
        for p in self.points:
            if is_visible(point, p, self.obstacles):
                # Every entry is a tuple (point, weight)
                weight = point.distance(p)
                self.visibility_graph.setdefault(point, []).append((weight, p))
                self.visibility_graph.setdefault(p, []).append((weight, point))

    # Generate the visibility map
    def generate_visibility_graph(self, k_neighbors=None):
        graph = {}

        # Function to check closest neighbors
        def k_nearest_neighbors(p, points, k):
            distances = [(p.distance(point), point) for point in points if point != p]
            heapq.heapify(distances)
            return [heapq.heappop(distances)[1] for _ in range(min(k, len(distances)))]

        for p1 in self.points:
            if k_neighbors is not None:  # If k_neighbors is not None, we only check the k closest neighbors
                neighbors = k_nearest_neighbors(p1, self.points, k_neighbors)
            else:  # Otherwise, we check all the points
                neighbors = [p for p in self.points if p != p1]

            # Check if the points are visible from each other
            for p2 in neighbors:
                if is_visible(p1, p2, self.obstacles):
                    # If they are we add them to the graph
                    weight = p1.distance(p2)
                    graph.setdefault(p1, []).append((weight, p2))
                    graph.setdefault(p2, []).append((weight, p1))
        return graph

    # A* algorithm
    def a_star(self, start, goal):
        open_set = [(0, start, [])]  # (priority, node, path)
        closed_set = set()

        while open_set:
            _, current, path = heapq.heappop(open_set)
            if current == goal:
                return path + [current]

            if current in closed_set:
                continue

            closed_set.add(current)
            path = path + [current]

            for weight, neighbor in self.visibility_graph[current]:
                if neighbor not in closed_set:
                    g_score = weight + path[-1].distance(current)
                    f_score = g_score + neighbor.distance(goal)
                    heapq.heappush(open_set, (f_score, neighbor, path))

        return None

    def plot(self, start, goal):
        # Add start and goal points to the visibility graph
        self.add_point_to_visibility_graph(start)
        self.add_point_to_visibility_graph(goal)

        # Initialize g_score and f_score for start and goal points
        path = self.a_star(start, goal)

        # Plot the obstacles and the visibility graph
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(grid, cmap='gray_r', origin='lower')
        for obstacle in self.obstacles:
            xs, ys = obstacle.exterior.xy
            ax.fill(xs, ys, color='k')
        for point in self.points:
            ax.scatter(point.x, point.y, color='b')
        self.plot_visibility_graph(ax=ax)

        # Plot the path
        if path is not None and len(path) > 1:
            for i in range(len(path) - 1):
                p1, p2 = path[i], path[i + 1]
                ax.plot([p1.x, p2.x], [p1.y, p2.y], 'g-', linewidth=3)

        return fig, ax

    def plot_visibility_graph(self, ax=None):
        graph = self.visibility_graph
        if ax is None:
            fig, ax = plt.subplots()

        for p1, neighbors in graph.items():
            for weight, p2 in neighbors:
                ax.plot([p1.x, p2.x], [p1.y, p2.y], 'r-', alpha=0.5)


class RRT:
    def __init__(self, start, goal, obstacles, map_size, max_iterations=1000, step_size=10):
        self.start = start
        self.goal = goal
        self.obstacles = obstacles
        self.map_size = map_size
        self.max_iterations = max_iterations
        self.step_size = step_size
        self.nodes = [start]
        self.edges = []

    def find_nearest_node(self, point):
        nearest_node = self.nodes[0]
        min_distance = nearest_node.distance(point)

        for node in self.nodes[1:]:
            distance = node.distance(point)
            if distance < min_distance:
                nearest_node = node
                min_distance = distance

        return nearest_node

    def is_collision_free(self, line):
        for obstacle in self.obstacles:
            if line.intersects(obstacle) and not line.touches(obstacle):
                return False
        return True

    def build_rrt(self):
        for _ in range(self.max_iterations):
            random_point = sg.Point(random.randint(0, self.map_size), random.randint(0, self.map_size))

            nearest_node = self.find_nearest_node(random_point)
            direction = np.array([random_point.x - nearest_node.x, random_point.y - nearest_node.y])
            normalized_direction = direction / np.linalg.norm(direction)
            step = normalized_direction * self.step_size
            new_node = sg.Point(nearest_node.x + step[0], nearest_node.y + step[1])

            if self.is_collision_free(sg.LineString([nearest_node, new_node])):
                self.nodes.append(new_node)
                self.edges.append((nearest_node, new_node))

                if new_node.distance(self.goal) <= self.step_size:
                    if self.is_collision_free(sg.LineString([new_node, self.goal])):
                        self.nodes.append(self.goal)
                        self.edges.append((new_node, self.goal))
                        return True
        return False

    def plot(self, start, goal):
        # Plot the result
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(grid, cmap='gray_r', origin='lower')
        for obstacle in self.obstacles:
            xs, ys = obstacle.exterior.xy
            ax.fill(xs, ys, color='k')
        for edge in self.edges:
            ax.plot([edge[0].x, edge[1].x], [edge[0].y, edge[1].y], 'r-', alpha=0.5)
        ax.scatter(start.x, start.y, color='b', label='Start')
        ax.scatter(goal.x, goal.y, color='g', label='Goal')
        ax.legend()
        return fig, ax


def generate_random_map(size, obstacles, min_vertices=3, max_vertices=7, min_size=5, max_size=20):
    # Create a grid of zeros
    grid = np.zeros((size, size), dtype=float)
    # Create a list of obstacles
    obstacles_list = []
    # Create the obstacles on the map
    for _ in range(obstacles):
        # Generate a random number of vertices for the polygon
        num_vertices = np.random.randint(min_vertices, max_vertices + 1)
        # Create random points for the polygon
        points = [np.random.rand(2) for _ in range(num_vertices)]
        # Compute the convex hull of the points
        hull = ConvexHull(points)
        # Create a random polygon
        obstacle = sg.Polygon([points[v] for v in hull.vertices])
        # Scale the polygon to a random size
        random_size = np.random.uniform(min_size, max_size)
        scaled_obstacle = affinity.scale(obstacle, xfact=random_size, yfact=random_size, origin=(0, 0))
        # Move the polygon to a random position within the map
        random_position = np.random.randint(0, size - int(random_size), size=(2,))
        translated_obstacle = affinity.translate(scaled_obstacle, xoff=random_position[0], yoff=random_position[1])
        # Add the obstacle to the list
        obstacles_list.append(translated_obstacle)
        # Add the obstacle to the grid
        for x in range(size):
            for y in range(size):
                if translated_obstacle.contains(sg.Point(x, y)):
                    grid[y, x] = 1
    # Return the grid and the obstacles
    return grid, obstacles_list


size = 50
num_points = 50
num_obstacles = 5

# Generate a random map
grid, obstacles_list = generate_random_map(size, num_obstacles)

# probabilistic version
probabilistic_map = ProberlisticMap(size, obstacles_list, num_points=num_points, max_checks=5)

# Do the same but with RRT
start = sg.Point(np.random.randint(0, size), np.random.randint(0, size))
goal = sg.Point(np.random.randint(0, size), np.random.randint(0, size))
rrt = RRT(start, goal, obstacles_list, size, max_iterations=1000, step_size=10)

# Generate the map
rrt.build_rrt()

# plot the result
fig1, ax1 = probabilistic_map.plot(start, goal)
fig2, ax2 = rrt.plot(start, goal)
plt.show()
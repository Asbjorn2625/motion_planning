import numpy as np
import matplotlib.pyplot as plt


def brushfire(configuration_space):
    """
    Brushfire function:
    Starts by creating a list of starting points, which is the edges and objects found in the array
    Then creates a queue wherein we walk to all neighbours and check if that area has been visited before
    If the neighbour is a new area the current distance from the starting point is saved
    we repeat this process till there are no more items in the queue
    """
    configuration_space=np.pad(configuration_space,pad_width=1, mode="constant")
    empty=np.zeros_like(configuration_space, dtype=int)
    empty[configuration_space != -1] = -1

    obstacles = set(zip(*np.where(configuration_space != -1)))  # Store obstacle cells as a set

    queue = list(obstacles)
    distance = 1
    while queue:
        size = len(queue)
        for _ in range(size):
            row, col = queue.pop(0)
            for d_row, d_col in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                neighbour = (row + d_row, col + d_col)
                if neighbour in obstacles or \
                        neighbour[0] < 0 or neighbour[0] >= empty.shape[0] or \
                        neighbour[1] < 0 or neighbour[1] >= empty.shape[1]:
                    continue  # Skip obstacle cells and cells outside the grid
                # Add the marked point into obstacles
                obstacles.add(neighbour)
                empty[neighbour[0], neighbour[1]] = distance
                queue.append(neighbour)
        distance += 1
    empty[empty == -1] = 0
    return empty


def find_voronoi(brushfire_array):
    """
    Voronoi graph function, found by looking at the increase of size in the brushfire array
    """
    voronoi = []
    for y in range(1, len(brushfire_array) - 1):
        for x in range(1, len(brushfire_array[y]) - 1):
            # By counting if the point is higher than least seven of it's neighbours we can figure out the voronoi
            count = 0
            # Neighbours are indexes to every side
            neighbours = [[0, 1], [0, -1], [1, 0], [-1, 0], [1, 1], [1, -1], [-1, 1], [-1, -1]]
            for neighbour in neighbours:
                y2, x2 = np.array(neighbour)+np.array([y, x])
                if brushfire_array[y, x] >= brushfire_array[y2, x2] and brushfire_array[y, x] != 0:
                    count += 1
            # Check if the index satisfies the condition
            if count >= 7:
                voronoi.append([y, x])
    return np.array(voronoi)




def main():
    configspace = np.load("config.pkl", allow_pickle=True)
    b=brushfire(configspace)
    indices = find_voronoi(b)
    v_x = indices[:, 0]
    v_y = indices[:, 1]
    plt.imshow(b, cmap='viridis')
    plt.scatter(v_y, v_x, c='red', s=10)
    plt.colorbar()
    plt.show()

if __name__=="__main__":
    main()
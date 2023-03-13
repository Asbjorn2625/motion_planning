import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label
from matplotlib.colors import ListedColormap


def connected_objects(input_array):
    mask = input_array >= 0
    # perform connected component analysis
    # Could have done this myself through a grassfire method, but I am lazy
    labeled_arr, num_features = label(mask)

    # create an array to store the sorted labels
    sorted_arr = np.zeros_like(labeled_arr)

    # sort the labels based on the size of the connected component
    for i in range(1, num_features + 1):
        sorted_arr[labeled_arr == i] = i

    # We only want to add the wall where there are no objects
    zero_entities = sorted_arr == 0
    # Set in wall points as an object
    sorted_arr[0][zero_entities[0]] = num_features + 2
    sorted_arr[-1][zero_entities[-1]] = num_features + 3
    sorted_arr[:, 0][zero_entities[:, 0]] = num_features + 4
    sorted_arr[:, -1][zero_entities[:, -1]] = num_features + 5

    return sorted_arr


def brushfire(configuration_space, group_array):
    """
    Brushfire function:
    Starts by creating a list of starting points, which is the edges and objects found in the array
    Then creates a queue wherein we walk to all neighbours and check if that area has been visited before
    If the neighbour is a new area the current distance from the starting point is saved
    we repeat this process till there are no more items in the queue
    """
    configuration_space[0] = 2
    configuration_space[-1] = 2
    configuration_space[:, 0] = 2
    configuration_space[:, -1] = 2
    # Create an output array
    brushfire_grid = np.zeros_like(configuration_space, dtype=int)
    # Fill in the objects, with an arbitrary number that we will not get near
    brushfire_grid[configuration_space != -1] = -1

    # Create an output array for the obstacle grid
    obstacle_grid = np.copy(group_array)

    obstacles = set(zip(*np.where(configuration_space != -1)))  # Store obstacle cells as a set
    # We find the group associated with the obstacle
    groups = [group_array[i, j] for i, j in obstacles]
    # Add the two into the queue
    queue = list(zip(obstacles, groups))
    distance = 1
    # Continue to loop through until the queue is empty
    while queue:
        size = len(queue)
        for _ in range(size):
            pos, group = queue.pop(0)
            row, col = pos
            for d_row, d_col in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                neighbour = (row + d_row, col + d_col)
                if neighbour in obstacles or \
                        neighbour[0] < 0 or neighbour[0] >= brushfire_grid.shape[0] or \
                        neighbour[1] < 0 or neighbour[1] >= brushfire_grid.shape[1]:
                    continue  # Skip obstacle cells and cells outside the grid
                # Add the marked point into obstacles
                obstacles.add(neighbour)
                # Write in the distance
                brushfire_grid[neighbour[0], neighbour[1]] = distance
                # Write in what object the distance derived from
                obstacle_grid[neighbour[0], neighbour[1]] = group
                # Append the new position into the queue
                queue.append([neighbour, group])
        # For every run through the queue we increase the distance by one
        distance += 1
    # Change the -1 to zero, such that anything inside an obstacle have 0 distance to one
    brushfire_grid[brushfire_grid == -1] = 0
    return brushfire_grid, obstacle_grid


def find_voronoi(obstacle_grid):
    import cv2
    """
    Voronoi graph function, found by looking at the obstacle grid
    """
    # Find edges of expanded obstacle grid
    edges = cv2.Canny(obstacle_grid.astype(np.uint8), 0, 0)

    return edges


def find_voronoi_brushfire(brushfire_array):
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



def create_bursh_voronoi_grid(plot=True):
    configspace = np.load("config.pkl", allow_pickle=True)
    # Find out which object is connected
    blobs = connected_objects(configspace)
    # Run the brushfire
    brush, object_grid = brushfire(configspace, blobs)

    # Voronoi from the obstacle grid
    vor = find_voronoi(object_grid)

    if plot:
        # Voronoi from the brushfire
        indices_brush = find_voronoi_brushfire(brush)
        vb_x = indices_brush[:, 0]
        vb_y = indices_brush[:, 1]
        # Plot the brushfire version
        plt.imshow(brush, cmap='viridis')
        plt.scatter(vb_y, vb_x)
        plt.colorbar()
        plt.show()

        # Plot the object grid alone
        plt.imshow(object_grid, cmap='viridis')
        plt.colorbar()
        plt.show()

        # plot the voronoi from the object grid
        vor_map = brush
        vor_map[vor != 0] = -1
        # Define the colormap
        colors = ["#440154", "#3e4a89", "#2a788e", "#22a884", "#7ebf41", "#fde725"]
        cmap = ListedColormap(colors)
        cmap.set_under('r')
        plt.imshow(vor_map, cmap=cmap, vmin=0)
        plt.colorbar()
        plt.show()
    return brush, vor

if __name__=="__main__":
    create_bursh_voronoi_grid()
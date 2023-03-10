import numpy as np
import matplotlib.pyplot as plt



def brushfire(configuration_space):
    configuration_space=np.pad(configuration_space,pad_width=1, mode="constant")
    empty=np.zeros_like(configuration_space)
    empty[configuration_space != -1] = -1

    startpoints= np.where(configuration_space!=-1)
    queer=[]
    for i in range(len(startpoints[0])):
        row=startpoints[0][i]
        col=startpoints[1][i]
        queer.append([row,col])
    distance = 1
    while queer:
        row,col = queer.pop(0)
        kernel = np.array([(row+1,col),(row-1,col),(row,col-1),(row,col+1)])
        for neighbour in kernel:
            if neighbour[0] < 0 or neighbour[0] >= empty.shape[0] or \
                neighbour[1] < 0 or neighbour[1] >= empty.shape[1]:
                continue
            if empty[neighbour[0],neighbour[1]] != 0:
                continue
            empty[neighbour[0],neighbour[1]]= distance
            queer.append(neighbour)
            distance += 1
    empty[empty == -1] = 0
    return empty


def find_voronoi_points(brushfire_array):

    # Calculate the distance to the nearest zero value along the rows and cols
    equal_dist_indices = []
    for i in range(brushfire_array.shape[0]):
        for j in range(brushfire_array.shape[1]):
            if brushfire_array[i, j] == 0:  # If we are on a wall, the distance is very short
                continue
            # Check the distances along the rows
                # Compute the distances to the nearest zero value in all four directions
            left_dist = np.abs(j - np.where(brushfire_array[i, :j] == 0)[0][-1]) if np.any(brushfire_array[i, :j] == 0) else np.inf
            right_dist = np.abs(j - np.where(brushfire_array[i, j + 1:] == 0)[0][0] + j + 1) if np.any(brushfire_array[i, j + 1:] == 0) else np.inf
            up_dist = np.abs(i - np.where(brushfire_array[:i, j] == 0)[0][-1]) if np.any(brushfire_array[:i, j] == 0) else np.inf
            down_dist = np.abs(i - np.where(brushfire_array[i + 1:, j] == 0)[0][0] + i + 1) if np.any(brushfire_array[i + 1:, j] == 0) else np.inf

            # Check if the distances to any zero value in the four directions are equal to another
            if (left_dist == right_dist) or (left_dist == up_dist) or (left_dist == down_dist) or (
                    right_dist == up_dist) or (right_dist == down_dist) or (up_dist == down_dist):
                equal_dist_indices.append([i, j])

    # Return only the indexes of where the edge is
    return equal_dist_indices




def main():
    configspace = np.load("config.pkl", allow_pickle=True)
    b=brushfire(configspace)
    indices = np.array(find_voronoi_points(b))
    # Extract the x and y coordinates from the input array
    x_coords = indices[:,1]
    y_coords = indices[:,0]

    # Create a scatter plot of the points
    plt.scatter(x_coords, y_coords, s=1)
    plt.imshow(b, cmap='viridis')
    plt.colorbar()
    plt.show()

if __name__=="__main__":
    main()
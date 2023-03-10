import pickle
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


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


def find_voronoi_points(original_array, brushfire_array):

    box_indices = np.nonzero(original_array != -1)
    # Loop through the locations of the collision boxes and check for edges
    edge_indices = []
    G  = nx.Graph()
    for i, j in zip(*box_indices):
        for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            ni, nj = i + di, j + dj
            if ni < 0 or ni >= original_array.shape[0] or \
                nj < 0 or nj >= original_array.shape[1]:
                continue
            if original_array[ni, nj] != -1:
                G.add_edge((i, j), (ni, nj))

    # Identify the connected components of the graph
    components = list(nx.connected_components(G))

    # Create a dictionary mapping each edge to its component index
    component_dict = {}
    for i, component in enumerate(components):
        for edge in component:
            component_dict[edge] = i

    component_dict[len(component_dict)+1] = [(0, i) for i in range(len(original_array.shape[0]))]
    component_dict[len(component_dict) + 2] = [(i, 0) for i in range(len(original_array.shape[0]))]
    component_dict[len(component_dict) + 3] = [(len(original_array.shape[0]), i) for i in range(len(original_array.shape[0]))]
    component_dict[len(component_dict) + 4] = [(i, len(original_array.shape[0])) for i in range(len(original_array.shape[0]))]



    return None




def main():
    configspace = np.load("config.pkl",allow_pickle=True)
    b=brushfire(configspace)
    c = find_voronoi_points(configspace, b)
    plt.imshow(b, cmap='viridis')
    plt.colorbar()
    plt.show()

if __name__=="__main__":
    main()
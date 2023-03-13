"""ur_cspace_explorer controller."""

import struct
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from controller import Supervisor, Motor, Receiver, PositionSensor
import pickle as pkl
from brush import *
import heapq

robot = Supervisor()

# Define plot
plot = True

TIME_STEP = int(robot.getBasicTimeStep())

shoulder_lift_motor = Motor("shoulder_lift_joint")
shoulder_lift_sensor = PositionSensor("shoulder_lift_joint_sensor")
elbow_motor = Motor("elbow_joint")
elbow_sensor = PositionSensor("elbow_joint_sensor")

radio_receiver = Receiver("receiver")
radio_receiver.enable(TIME_STEP)

# Set the mapping accuracy
angle_tick = 0.05
shoulder_min_angle = -3.14
shoulder_max_angle = 0
elbow_min_angle = -2.4
elbow_max_angle = 2.4
shoulder_positions = np.arange(shoulder_min_angle, shoulder_max_angle, angle_tick)
elbow_positions = np.arange(elbow_min_angle, elbow_max_angle, angle_tick)


def bug_movement(start_pos, end_pos, collision):
    """
    Bug algortihm for moving the robot
    """

    set_position_sync(shoulder_lift_motor, shoulder_lift_sensor, start_pos[0])
    set_position_sync(elbow_motor, elbow_sensor, start_pos[1])
    
    # Initialize position to starting point
    position = start_pos
    
    # Get the shortest time
    tf = np.sqrt((end_pos[0]-start_pos[0])**2+(end_pos[1]-start_pos[1])**2)
    
    para_shoulder = parabolic_blend(start_pos[0], end_pos[0], tf)
    para_elbow = parabolic_blend(start_pos[1], end_pos[1], tf)
    # set starting time
    ss = robot.getTime()
    
    while True:
        current_time = robot.getTime() - ss
        if current_time >= tf:
            break
        position = (parabolic_return(para_shoulder, current_time), parabolic_return(para_elbow, current_time))
        
        shoulder_closest_pos = np.abs(shoulder_positions - position[0]).argmin()
        elbow_closest_pos = np.abs(elbow_positions - position[1]).argmin()

        print(shoulder_closest_pos)

        if collision[shoulder_closest_pos][elbow_closest_pos] == -1:
            print('we sending')
            set_position_sync(shoulder_lift_motor, shoulder_lift_sensor, shoulder_closest_pos)
            set_position_sync(elbow_motor, elbow_sensor, elbow_closest_pos)
        else:
            print('hit')
            return

def safe_move(s_pos, e_pos, voronoi_map):
    # Get the position of the robot
    shoulder_joint_pos = s_pos[0]
    elbow_joint_pos = s_pos[1]

    # Find the closest safe position in the Voronoi map for the shoulder joint
    shoulder_closest_pos = shoulder_positions[np.argmin(np.abs(shoulder_positions - shoulder_joint_pos))]
    shoulder_index = int((shoulder_closest_pos - shoulder_min_angle) / angle_tick)

    # Find the closest safe position in the Voronoi map for the elbow joint
    elbow_closest_pos = elbow_positions[np.argmin(np.abs(elbow_positions - elbow_joint_pos))]
    elbow_index = int((elbow_closest_pos - elbow_min_angle) / angle_tick)

    # Get the coordinates of the given position
    start_pos = (shoulder_index, elbow_index)

    # Do the same for the end position
    shoulder_joint_pos = e_pos[0]
    elbow_joint_pos = e_pos[1]

    shoulder_closest_pos = shoulder_positions[np.argmin(np.abs(shoulder_positions - shoulder_joint_pos))]
    shoulder_index = int((shoulder_closest_pos - shoulder_min_angle) / angle_tick)
    elbow_closest_pos = elbow_positions[np.argmin(np.abs(elbow_positions - elbow_joint_pos))]
    elbow_index = int((elbow_closest_pos - elbow_min_angle) / angle_tick)

    # Get the coordinates of the given position
    end_pos = (shoulder_index, elbow_index)

    # Identify the safe path in the array and their indices
    safe_values, safe_indices = np.where(voronoi_map != 0), np.argwhere(voronoi_map != 0)

    # Calculate the Euclidean distance between the given position and all positive values
    start_distances = np.linalg.norm(np.array(safe_values).T - np.array(start_pos), axis=1)
    end_distances = np.linalg.norm(np.array(safe_values).T - np.array(end_pos), axis=1)

    # Find the starting position of the safe path
    closest_start_index = safe_indices[np.argmin(start_distances)]
    # Find the ending position of the safe path
    closest_end_index = safe_indices[np.argmin(end_distances)]

    # Walk to the starting position of the safe path
    set_position_sync(shoulder_lift_motor, shoulder_lift_sensor, shoulder_positions[closest_start_index[0]])
    set_position_sync(elbow_motor, elbow_sensor, elbow_positions[closest_start_index[1]])

    # Find the path with Dijkstras
    path = dijkstras_algorithm(voronoi_map, tuple(closest_start_index), tuple(closest_end_index))
    for pos in path:
        set_position_sync(shoulder_lift_motor, shoulder_lift_sensor, shoulder_positions[pos[0]])
        set_position_sync(elbow_motor, elbow_sensor, elbow_positions[pos[1]])
    # Go to the actual ending position
    set_position_sync(shoulder_lift_motor, shoulder_lift_sensor, e_pos[0])
    set_position_sync(elbow_motor, elbow_sensor, e_pos[1])

# ------- Dijkstras algorithm ---------------------------


def dijkstras_algorithm(graph, start, end):
    """
    Implements Dijkstra's algorithm to find the shortest path
    between the start and end points in a NumPy array graph.
    """
    # Define the neighbors of a point as those within a Chebyshev distance of 3
    def get_neighbors(point):
        neighbors = []
        for row_offset in range(-3, 4):
            for col_offset in range(-3, 4):
                if row_offset == 0 and col_offset == 0:
                    continue
                neighbor_row = point[0] + row_offset
                neighbor_col = point[1] + col_offset
                if (0 <= neighbor_row < graph.shape[0] and
                        0 <= neighbor_col < graph.shape[1] and
                        graph[neighbor_row, neighbor_col] != 0):
                    neighbors.append((neighbor_row, neighbor_col))
        return neighbors
    
    # Initialize the distances to all points as infinity, except the start point
    distances = np.full(graph.shape, np.inf)
    distances[start[0], start[1]] = 0
    
    # Initialize the priority queue with the start point
    queue = [(0, start)]
    
    # Initialize the previous nodes to None
    previous = np.full(graph.shape, None, dtype=np.object)
    
    # Iterate until the priority queue is empty or the end point is reached
    while queue:
        # Get the point with the minimum distance
        current_distance, current_point = heapq.heappop(queue)
        
        # If this is the end point, stop iterating
        if current_point == end:
            break
        
        # Iterate over the neighbors of the current point
        for neighbor_point in get_neighbors(current_point):
            # Calculate the distance to the neighbor through the current point
            neighbor_distance = current_distance + np.sqrt((neighbor_point[0] - current_point[0])**2 + (neighbor_point[1] - current_point[1])**2)
            
            # If this is a shorter distance than previously calculated, update the distance
            if neighbor_distance < distances[neighbor_point]:
                distances[neighbor_point] = neighbor_distance
                
                # Update the previous node for this neighbor
                previous[neighbor_point] = current_point
                
                # Add the neighbor to the priority queue
                heapq.heappush(queue, (neighbor_distance, neighbor_point))
    
    # If the end point was not reached, return an empty path
    if previous[end] is None:
        return []
    
    # Build the path from the previous nodes
    path = []
    current_point = end
    while current_point is not None:
        path.append(current_point)
        current_point = previous[current_point]
    path.reverse()
    
    return path

# --------------------------------------------------------
def parabolic_return(a, t):
    return a[0]+a[1]*t+a[2]*t**2+a[3]*t**3
    
def parabolic_blend(start, end, t):
    """Calculate coefficients for a parabolic blend between start and end points over time t."""
    distance = end - start
    a0 = start
    a1 = 0
    a2 = 3 * distance / t ** 2
    a3 = -2 * distance / t ** 3
    return a0, a1, a2, a3
  
    
def heading_difference(start_point, end_point):
    """Get the angle difference between the robot heading and a line

    The line is defined by a start and an end point. The angle between the
    heading and the line is given by the dot product. To get orientation
    we need to look at the direction of the cross product along the z-axis.
    """
    rot_robot = np.array(robot.getSelf().getOrientation()).reshape(3, 3)
    robot_heading = np.dot( rot_robot, np.array([1,0,0]) )
    line = (np.array(end_point) - np.array(start_point))
    angle = acos( np.dot(robot_heading, line)
                  / np.linalg.norm(line)
                  / np.linalg.norm(robot_heading) )
    cross = np.cross(line, robot_heading)
    if cross[2] > 0.0:
        angle = -angle
    
    return angle


def set_position_sync(a_motor, a_sensor, target, tolerance=0.001):
    """Move joint to a set position while waiting for the result

    Arguments
    ---------
    a_motor
        The motor to move.
    a_sensor
        The position sensor coresponding to the motor.
    target
        The position you want to go to.
    tolerance
        How close to the target do we want to be?
    """
    a_sensor.enable(TIME_STEP)
    a_motor.setPosition(target)
    robot.step(TIME_STEP)  # simulate one step to get sensor data
    effective = a_sensor.getValue()  # effective position
    while abs(target - effective) > tolerance:
        if robot.step(TIME_STEP) == -1:
            break
        effective = a_sensor.getValue()
    a_sensor.disable()


def check_collision():
    """Check collisions between the special DEFed WALL# elements.

    The physics plugin "no_collision_walls" expects solids named WALL#, where
    # is sequential numbers starting from 0. This function uses a fake-radio
    receiver to get the wall that is being intersected.
    """
    
    if radio_receiver.getQueueLength() > 0:
        while radio_receiver.getQueueLength() > 1:
            radio_receiver.nextPacket()
        message = radio_receiver.getInts()

        radio_receiver.nextPacket()
        
        return message[0]
    return 0


def get_configspace():
    """Example that checks the configuration space for two revolute joints."""
    # Explore the C space
    contact = np.zeros((len(shoulder_positions), len(elbow_positions)))
    for shoulder_i, shoulder_pos in enumerate(shoulder_positions):
        set_position_sync(shoulder_lift_motor, shoulder_lift_sensor, shoulder_pos)
        for elbow_i, elbow_pos in enumerate(elbow_positions):
            set_position_sync(elbow_motor, elbow_sensor, elbow_pos)
            mes = check_collision()
            if check_collision == 0:
                continue
            contact[shoulder_i][elbow_i] = mes
    if plot:
        # Plot it
        num_walls = int(np.amax(contact))
        fig, axes = plt.subplots()
        color_map = cm.get_cmap("Blues", num_walls + 1)
        colors = plt.imshow(contact, cmap=color_map)
        cbar = fig.colorbar(colors, ax=axes)
        cbar.ax.get_yaxis().set_ticks([])
        for j in range(num_walls + 1):
            cbar.ax.text(5, (j + 0.5) / (1 + 1 / num_walls), j, ha="center", va="center")
        plt.xlabel("Elbow angle")
        plt.xticks(
            np.linspace(0, elbow_positions.size, 5),
            ["{:.2f}".format(n) for n in np.linspace(elbow_min_angle, elbow_max_angle, 5)],
        )
        plt.ylabel("Shoulder angle")
        plt.yticks(
            np.linspace(0, shoulder_positions.size, 5),
            ["{:.2f}".format(n) for n in np.linspace(shoulder_min_angle, shoulder_max_angle, 5)],
        )
        plt.show()
    return contact


if __name__ == "__main__":
    # Get the configspace
    config = get_configspace()
    # Save the config space as pickle
    with open("config.pkl", "wb") as f:
        pkl.dump(config, f)
    # Get the voronoi graph
    brush, voronoi = create_bursh_voronoi_grid(plot=plot)

    # Map out start and goal positions
    start_pos = (0, -2.4)
    end_pos = (-2, 2.30)

    # Run through the voronoi graph
    safe_move(start_pos, end_pos, voronoi)
    #bug_movement(start_pos, end_pos, config)
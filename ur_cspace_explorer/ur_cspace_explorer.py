"""ur_cspace_explorer controller."""

import struct
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from controller import Supervisor, Motor, Receiver, PositionSensor
import pickle as pkl

robot = Supervisor()

TIME_STEP = int(robot.getBasicTimeStep())

shoulder_lift_motor = Motor("shoulder_lift_joint")
shoulder_lift_sensor = PositionSensor("shoulder_lift_joint_sensor")
elbow_motor = Motor("elbow_joint")
elbow_sensor = PositionSensor("elbow_joint_sensor")

radio_receiver = Receiver("receiver")
radio_receiver.enable(TIME_STEP)


def bug_movement(start_pos, end_pos, collision):
    """
    Bug algortihm for moving the robot
    """
    # TODO: convert values to the i postion in the NP array
    angle_tick = 0.1
    shoulder_min_angle = -3.14
    shoulder_max_angle = 0
    elbow_min_angle = -2.4
    elbow_max_angle = 2.4
    
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
        if collision[position[0]][position[1]] == 0:
            set_position_sync(shoulder_lift_motor, shoulder_lift_sensor, position[0])
            set_position_sync(elbow_motor, elbow_sensor, position[1])
        else:
            print('hit')
        
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
    angle_tick = 0.05
    shoulder_min_angle = -3.14
    shoulder_max_angle = 0
    elbow_min_angle = -2.4
    elbow_max_angle = 2.4
    shoulder_postitions = np.arange(shoulder_min_angle, shoulder_max_angle, angle_tick)
    elbow_postitions = np.arange(elbow_min_angle, elbow_max_angle, angle_tick)
    contact = np.zeros((len(shoulder_postitions), len(elbow_postitions)))
    for shoulder_i, shoulder_pos in enumerate(shoulder_postitions):
        set_position_sync(shoulder_lift_motor, shoulder_lift_sensor, shoulder_pos)
        for elbow_i, elbow_pos in enumerate(elbow_postitions):
            set_position_sync(elbow_motor, elbow_sensor, elbow_pos)
            mes = check_collision()
            if check_collision == 0:
                continue
            contact[shoulder_i][elbow_i] = mes

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
        np.linspace(0, elbow_postitions.size, 5),
        ["{:.2f}".format(n) for n in np.linspace(elbow_min_angle, elbow_max_angle, 5)],
    )
    plt.ylabel("Shoulder angle")
    plt.yticks(
        np.linspace(0, shoulder_postitions.size, 5),
        ["{:.2f}".format(n) for n in np.linspace(shoulder_min_angle, shoulder_max_angle, 5)],
    )
    plt.show()
    return contact


if __name__ == "__main__":
    config = get_configspace()
    with open("config.pkl", "wb") as f:
        pkl.dump(config, f)
    start_pos = (-0.85, -1.2)
    end_pos = (-1.20, 1.40)
    bug_movement(start_pos, end_pos, config)
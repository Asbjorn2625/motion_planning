import numpy as np
import cv2
from shapely.geometry import Point, LineString

# Generate 50 random points
points = np.random.rand(100, 2)

Startpoint = points[np.random.randint(0, 99)]
Endpoint = points[np.random.randint(0, 99)]

# Create Point objects from the points using Shapely
point_objs = [Point(x) for x in points]

# amount of obstacles
obstacle_amount = np.random.randint(1, 4)
obstacle_type = np.random.randint(0, 2, size=obstacle_amount)
print(obstacle_type)

# Create an empty image
img = np.zeros((500, 500, 3), dtype=np.uint8)

# Scale the points to fit the image size
points[:, 0] *= img.shape[1]
points[:, 1] *= img.shape[0]

# Draw circles at the locations of the points
for point in points:
    center = (int(point[0]), int(point[1]))
    cv2.circle(img, center, 3, (255, 255, 255), -1)

#Draw obstacles on the image
obstacle_color = (125, 125, 125)
obstacle_objects = []
for obstacle in obstacle_type:
    # Get the type and color of the obstacle
    if obstacle == 0:
        # Define the size and position of the rectangle
        size = (50, 50)
        pos = (np.random.randint(0, 500 - size[0]), np.random.randint(0, 500 - size[1]))
        # Draw the rectangle on the image
        cv2.rectangle(img, pos, (pos[0] + size[0], pos[1] + size[1]), obstacle_color, -1)
        obstacle_objects.append(LineString([(pos[0], pos[1]), (pos[0] + size[0], pos[1]), 
                                            (pos[0] + size[0], pos[1] + size[1]), (pos[0], pos[1] + size[1]), 
                                            (pos[0], pos[1])]))
    elif obstacle == 1:
        # Define the size and position of the circle
        radius = 50
        center = (np.random.randint(radius, 500 - radius), np.random.randint(radius, 500 - radius))
        # Draw the circle on the image
        cv2.circle(img, center, radius, obstacle_color, -1)
        obstacle_objects.append(Point(center).buffer(radius))

# Draw red lines between points that are not obstructed
line_color = (0, 0, 255)
for i in range(len(points)):
    for j in range(i+1, len(points)):
        line = LineString([points[i], points[j]])
        is_obstructed = False
        for obstacle in obstacle_objects:
            if line.intersects(obstacle):
                is_obstructed = True
                break
        if not is_obstructed:
            cv2.line(img, (int(points[i][0]), int(points[i][1])), (int(points[j][0]), int(points[j][1])), line_color, 1)

# Display the image
cv2.imshow("Hammersley Points", img)
cv2.waitKey(0)
cv2.destroyAllWindows()


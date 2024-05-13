import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image

def recover_z(depth_map):
    max = 271.4
    min = 2.46
    depth = depth_map / 255.0 * (max - min) + min
    depth = max - depth + 1000
    return depth

def image_to_point_cloud(image, focal_length):
    # Convert image to grayscale
    gray_image = image.convert('L')

    # Convert image to numpy array
    depth_map = np.array(gray_image)
    depth_map = recover_z(depth_map)

    # Compute the point cloud
    height, width = depth_map.shape
    cx, cy = width // 2, height // 2

    # Create a 3D array to hold the point cloud data
    point_cloud = []

    # Compute the x, y, and z coordinates
    for i in range(height):
        for j in range(width):
            z = depth_map[i, j]
            x = (j - cx) * z / focal_length
            y = (i - cy) * z / focal_length
            point_cloud.append([x, y, z])

    return point_cloud

def write_ply(filename, points):
    with open(filename, 'w') as f:
        # Write the header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("element vertex {}\n".format(len(points)))
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")

        # Write the points
        for point in points:
            f.write("{} {} {}\n".format(point[0], point[1], point[2]))

if __name__ == '__main__':
    # Load image
    image_path = 'output/2-frame-079_depth.png'  # Replace with your image path
    image = Image.open(image_path)

    # Focal length
    focal_length = 500  # Adjust as needed

    # Convert image to point cloud
    point_cloud = image_to_point_cloud(image, focal_length)

    write_ply('output/2-frame-079_depth.ply', point_cloud)

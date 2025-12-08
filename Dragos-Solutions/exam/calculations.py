import sympy as sp
import numpy as np

def sum_of_squared_differences(matrix1,matrix2):
    return np.sum(np.square(matrix1 - matrix2))

def sharpening(matrix1, matrix2, alpha):
    return (1 + alpha) * matrix1 - (alpha * 1/9) * matrix2

def size_of_image(x,y,pixel_range):
    return x*y*(int(pixel_range).bit_length())

def camera_projection_matrix(fx, fy, cx, cy):
    return np.matrix([[fx, 0, cx],
                      [0, fy, cy],
                      [0, 0, 1]])

def calculate_xy_camera_coordinates(camera_intrinsics, world_coordinates, rotation_matrix=np.eye(3), translation_vector=np.zeros((3, 1))):
    # Ensure world_coordinates is a column vector
    if world_coordinates.shape == (1, 3):
        world_coordinates = world_coordinates.T
    elif world_coordinates.shape == (3,):
        world_coordinates = world_coordinates.reshape(3, 1)

    world_homogeneous = np.vstack([world_coordinates, [[1]]])

    extrinsic_matrix = np.hstack([rotation_matrix, translation_vector.reshape(3, 1)])

    pixel_homogeneous = camera_intrinsics @ extrinsic_matrix @ world_homogeneous

    pixel_coords = pixel_homogeneous[:2] / pixel_homogeneous[2]

    return pixel_coords.flatten()


def initial_mean_shift_clusters(height, width):
    return height * width


# For your specific case
height = 480
width = 640
initial_clusters = initial_mean_shift_clusters(height, width)
print(f"Initial clusters: {initial_clusters}")

SSD1 = np.matrix([[10,15,20],
                    [20,20,25],
                    [10,15,20]])
SSD2 = np.matrix([[15,15,15],
                    [20,20,20],
                    [30,30,30]])

print(sum_of_squared_differences(SSD1,SSD2))

matrix1 = np.matrix([[0,0,0],[0,1,0],[0,0,0]])
matrix2 = np.ones((3,3))


print(sharpening(matrix1, matrix2, 0.9))

print(size_of_image(400,570,1023))

camera_intrinsics = camera_projection_matrix(725, 726, 631,360)
pixel_coordinates = np.matrix([1,1,5])
print(calculate_xy_camera_coordinates(camera_intrinsics, pixel_coordinates))
# print(int(1024).bit_length())

import numpy as np
from numpy.linalg import inv
import read_depth as rd
import cv2


def fuse_depth_values(z1, z2, x, P, A, B, C, Q, R):
    # Prediction step
    x = A.dot(x) + B.dot(u)
    P = A.dot(P).dot(A.T) + Q
    # Kalman gain
    S = C.dot(P).dot(C.T) + R
    K = P.dot(C.T).dot(inv(S))

    # Update step
    z = np.array([[z1],
                  [z2]])
    y = z - C.dot(x)
    x = x + K.dot(y.T)
    P = (np.eye(2) - K.dot(C)).dot(P)

    # Get fused depth estimate
    fused_depth_estimate = x[0,0]

    # Return fused depth estimate
    return fused_depth_estimate


# Generate some example depth maps
depth_stereo = rd.depth_read(r"D:\dataspell project\FinalYearProject\FinalYearProject\2011_09_26_drive_0001_syncimage_02\depth maps generated\0000000008.png") # Measured depth map from Stereo Camera
depth_lidar = rd.depth_read(r"D:\dataspell project\FinalYearProject\FinalYearProject\2011_09_26_drive_0001_syncimage_02\lidar ground truth\0000000008.png") # Measured depth map from LIDAR

# System matrices
A = np.array([[1, 0], # State transition matrix
              [0, 1]])
B = np.array([[0], # Control matrix
              [0]])
C = np.array([[1, 0]]) # Measurement matrix (only measuring depth)
Q = np.array([[0.5, 0],# Process noise covariance (how much do we trust the model)
              [0, 0.04]])
R = np.array([[0.1]]) # Measurement noise covariance (how much do we trust the measurements)
u = np.array([[0]]) # Control vector

# Initial state and covariance
x = np.array([[0], # Initial depth estimate
              [0]])
P = np.array([[1, 0], #` Initial covariance
              [0, 1]])

# Create empty depth estimate array
depth_estimate = np.zeros_like(depth_stereo)

# Loop through each pixel in the input depth maps and run the EKF algorithm
for i in range(depth_stereo.shape[0]):
    for j in range(depth_stereo.shape[1]):
        depth_estimate[i,j] = fuse_depth_values(depth_stereo[i,j], depth_lidar[i,j], x, P, A, B, C, Q, R)

# Save output image
depth_uint16 =  (depth_estimate * 256.0).astype(np.uint16)
cv2.imwrite(r"D:\dataspell project\FinalYearProject\FinalYearProject\2011_09_26_drive_0001_syncimage_02\Fused depth maps\0000000008test.png",depth_uint16)

print(depth_estimate)


# this code works with the disparity map produced by the SGBM algorithm and it shows the output of our depth map
# and compares it to the ground truth depth map
import cv2
import numpy as np
import read_depth as rd

# Matched block size. It must be an odd number >=1 . Normally, it should be somewhere in the 3..11 range.
block_size = 11
min_disp = -128
max_disp = 128
# Maximum disparity minus minimum disparity. The value is always greater than zero.
# In the current implementation, this parameter must be divisible by 16.
num_disp = max_disp - min_disp
# Margin in percentage by which the best (minimum) computed cost function value should "win" the second best value to consider the found match correct.
# Normally, a value within the 5-15 range is good enough
uniquenessRatio = 5
# Maximum size of smooth disparity regions to consider their noise speckles and invalidate.
# Set it to 0 to disable speckle filtering. Otherwise, set it somewhere in the 50-200 range.
speckleWindowSize = 200
# Maximum disparity variation within each connected component.
# If you do speckle filtering, set the parameter to a positive value, it will be implicitly multiplied by 16.
# Normally, 1 or 2 is good enough.
speckleRange = 2
disp12MaxDiff = 0

# Load the left and right stereo images
left_image = cv2.imread(r"left_rect.png", cv2.IMREAD_GRAYSCALE)
right_image = cv2.imread(r"right_rect.png", cv2.IMREAD_GRAYSCALE)
truth = rd.depth_read(r"D:\dataspell project\FinalYearProject\FinalYearProject\2011_09_26_drive_0001_syncimage_02\groundtruth\0000000015.png") # function for reading depth  ground truth depth map
# this line convert ground truth from uint16 to float32

# the above line is to convert the ground truth to the same scale as the disparity map

truth_depth=truth[147,247]
# here we are taking the depth value of a pixel in the ground truth image
cv2.imshow("left", left_image)
# Create a window to display the depth map
cv2.namedWindow("disp")

# Create trackbars for numDisparities and blockSize
cv2.createTrackbar('numDisparities','disp',1,31,lambda x: x)
cv2.createTrackbar('blockSize','disp',1,50,lambda x: None)
#cv2.createTrackbar('preFilterType','disp',1,1,lambda x: None)
#cv2.createTrackbar('preFilterSize','disp',2,25,lambda x: None)
cv2.createTrackbar('preFilterCap','disp',5,62,lambda x: None)
cv2.createTrackbar('textureThreshold','disp',10,100,lambda x: None)
cv2.createTrackbar('uniquenessRatio','disp',15,100,lambda x: None)
cv2.createTrackbar('speckleRange','disp',0,100, lambda x: None)
cv2.createTrackbar('speckleWindowSize','disp',3,25,lambda x: None)
cv2.createTrackbar('disp12MaxDiff','disp',5,25,lambda x: None)
cv2.createTrackbar('minDisparity','disp',5,25,lambda x: None)

while True:
    if cv2.getWindowProperty("disp", cv2.WND_PROP_VISIBLE) >= 0:
        # Get the current values of the trackbars
        numDisparities = cv2.getTrackbarPos('numDisparities','disp')*16
        blockSize = cv2.getTrackbarPos('blockSize','disp')*2 + 5
        preFilterType = cv2.getTrackbarPos('preFilterType','disp')
        preFilterSize = cv2.getTrackbarPos('preFilterSize','disp')*2 + 5
        preFilterCap = cv2.getTrackbarPos('preFilterCap','disp')
        textureThreshold = cv2.getTrackbarPos('textureThreshold','disp')
        uniquenessRatio = cv2.getTrackbarPos('uniquenessRatio','disp')
        speckleRange = cv2.getTrackbarPos('speckleRange','disp')
        speckleWindowSize = cv2.getTrackbarPos('speckleWindowSize','disp')*2
        disp12MaxDiff = cv2.getTrackbarPos('disp12MaxDiff','disp')
        minDisparity = cv2.getTrackbarPos('minDisparity','disp')

        # Calculate the depth map using the stereoBM algorithm
        stereo = cv2.StereoBM_create(numDisparities=4*16, blockSize=(3*(2+5)))
        stereo2=cv2.StereoSGBM_create(minDisparity=minDisparity, numDisparities=numDisparities, blockSize=blockSize, P1=8*3*blockSize**2, P2=32*3*blockSize**2, disp12MaxDiff=disp12MaxDiff, uniquenessRatio=uniquenessRatio, speckleWindowSize=speckleWindowSize, speckleRange=speckleRange, preFilterCap=preFilterCap,  mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
        stereo3=cv2.StereoSGBM_create(minDisparity=1, numDisparities=64, blockSize=11, P1=8*3*11**2, P2=32*3*11**2, disp12MaxDiff=4, uniquenessRatio=0, speckleWindowSize=0, speckleRange=0, preFilterCap=0,  mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

        disparity = stereo2.compute(left_image, right_image)  # here the disparity map is computed in pixels so
        # we need to convert it to meters using the focal length and the baseline distance
        # Converting to float32
      #  print(minDisparity, numDisparities, blockSize, preFilterCap, textureThreshold, uniquenessRatio, speckleRange, speckleWindowSize, disp12MaxDiff)
        disparity = disparity.astype(np.float32)
        real_disparity = disparity / 16.0
        # Convert the disparity map to a depth map using the baseline and focal length of the KITTI camera
        baseline = 0.537 # in meters
        focal_length = 721.5377 # in pixels
        depth = (focal_length * baseline) / real_disparity
        inf_indices = np.isinf(depth)
        depth[inf_indices] = 0
        print("this is the depth at 150x600 from the depth map",depth[147,247])
        print("this is the real depth at 150x600 ",truth_depth)
        depth=np.where(depth>256,255,depth)
        depth_uint16 =  (depth * 256.0).astype(np.uint16)
        depth = cv2.convertScaleAbs(depth)

    # Display the depth map
    # Apply a color map to the depth map
    color_map = cv2.applyColorMap(depth, cv2.COLORMAP_JET)

    # Display the depth map
    cv2.imshow("disp", color_map)

    # Exit if the user presses the 'Esc' key
    if cv2.waitKey(1) == 27:
        cv2.imwrite(("depth_map.png"), depth)
        cv2.imwrite(r"D:\dataspell project\FinalYearProject\FinalYearProject\2011_09_26_drive_0001_syncimage_02\depth maps generated\0000000015.png",depth_uint16)
        break

cv2.normalize(disparity, disparity, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)

cv2.imshow("disp", disparity)

if cv2.waitKey(1) & 0xFF == ord('q'):

    cv2.destroyAllWindows()
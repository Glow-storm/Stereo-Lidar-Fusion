
import customtkinter as ctk
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.figure import Figure
import pyzed.sl as sl
import math
import numpy as np
import cv2
import os
import ydlidar
import time
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from numpy.linalg import inv
new_width = 672   # 1280 for 720p, 2560 for 1080p, 672 for VGA


# Create empty array with the desired dimensions
new_arr = np.zeros((376, new_width))   # 720 for 720p, 1080 for 1080p, 376 for VGA

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


class LidarMainWindow(ctk.CTk):

    def __init__(self):
        super().__init__()

        # Set window title
        self.title('YDLidar LIDAR Monitor')

        # Create main widget
        self.widget = ctk.CTkFrame(self, bg="#1E2749")  # Added background color
        self.widget.pack(fill=ctk.BOTH, expand=True)

        # Create container for LIDAR plot and ZED camera image
        self.container = ctk.CTkFrame(self.widget, bg="#1E2749")  # Added background color
        self.container.pack(side=ctk.LEFT, fill=ctk.BOTH, expand=True)

        # Create figure and canvas for LIDAR plot
        self.fig = Figure(figsize=(4, 4), dpi=100)
        self.lidar_polar = self.fig.add_subplot(111, polar=True)
        self.lidar_polar.set_rmax(6)
        self.lidar_polar.grid(True)
        self.lidar_polar.set_title('LIDAR Plot')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.container)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=ctk.LEFT, fill=ctk.BOTH, expand=True)

        # Create separator
      #  self.separator = ctk.CTkFrame(self.container, bg="white")  # Added separator
       # self.separator.pack(side=ctk.LEFT, fill=ctk.Y, padx=10, pady=10)

        # Create label for ZED camera image description
        self.image_description_label = ctk.CTkLabel(self.container, text="Stereo Depth Map", bg="#1E2749", fg="white")
        self.image_description_label.config(font=("Arial", 14))
        self.image_description_label.pack(side=ctk.TOP, fill=ctk.X)  # Added font and color options

        # Create label and pixmap for ZED camera image
        self.image_container = ctk.CTkFrame(self.container, bg="#1E2749")  # Added background color
        self.image_container.pack(side=ctk.LEFT, fill=ctk.BOTH, expand=True)

        self.image_label = ctk.CTkLabel(self.image_container, width=672, height=240)
        self.image_label.pack(side=ctk.LEFT, fill=ctk.BOTH)

        self.image_description_label1 = ctk.CTkLabel(self.container, text="Fused Depth Map", bg="#1E2749", fg="white")
        self.image_description_label1.config(font=("Arial", 14))
        self.image_description_label1.pack(side=ctk.TOP, fill=ctk.X)  # Added font and color options

        # Create label and pixmap for ZED camera image
        self.image_container1 = ctk.CTkFrame(self.container, bg="#1E2749")  # Added background color
        self.image_container1.pack(side=ctk.LEFT, fill=ctk.BOTH, expand=True)

        self.image_label1 = ctk.CTkLabel(self.image_container1, width=672, height=240)
        self.image_label1.pack(side=ctk.LEFT, fill=ctk.BOTH)


        # Connect to YDLidar and ZED camera
        self.init_ydlidar()
        self.init_zed()

        # Start the timer to update the plot and image
        self.timer = self.after(0, self.update_lidar_and_image)

    def init_ydlidar(self):
        # Connect to YDLidar
        self.laser = ydlidar.CYdLidar()
        ports = ydlidar.lidarPortList()
        port = "/dev/ydlidar"
        for key, value in ports.items():
            port = value
        self.laser.setlidaropt(ydlidar.LidarPropSerialPort, port)
        self.laser.setlidaropt(ydlidar.LidarPropSerialPort, port)
        self.laser.setlidaropt(ydlidar.LidarPropSerialBaudrate, 115200)
        self.laser.setlidaropt(ydlidar.LidarPropLidarType, ydlidar.TYPE_TOF);
        self.laser.setlidaropt(ydlidar.LidarPropDeviceType, ydlidar.YDLIDAR_TYPE_SERIAL);
        self.laser.setlidaropt(ydlidar.LidarPropScanFrequency, 5);
        self.laser.setlidaropt(ydlidar.LidarPropSampleRate, 4);
        self.laser.setlidaropt(ydlidar.LidarPropSingleChannel, True);
        self.laser.setlidaropt(ydlidar.LidarPropMaxAngle, 180.0);
        self.laser.setlidaropt(ydlidar.LidarPropMinAngle, -180.0);

        ret = self.laser.initialize()
        ret = self.laser.turnOn();
        if not ret:
            print('Failed to connect to YDLidar')
            self.close()

    def init_zed(self):
        # Connect to ZED camera
        init = sl.InitParameters()
        init.depth_mode = sl.DEPTH_MODE.NEURAL  # Use PERFORMANCE depth mode
        init.coordinate_units = sl.UNIT.METER
        init.camera_resolution = sl.RESOLUTION.VGA  # Use HD720 video mode (default fps: 60)
        init.camera_fps = 60
        self.cam = sl.Camera()
        if not self.cam.is_opened():
            print("but why")
        #  print('Failed to open ZED camera')
        # self.close()
        status =self.cam.open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            print('Failed to open ZED camera')
            self.close()

    def fuse_depth_values(self,z1, z2, x, P, A, B, C, Q, R):
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

    def update_lidar_and_image(self):
        # Get current YDLidar data
        scan = ydlidar.LaserScan()
        data = self.laser.doProcessSimple(scan)
        if data:
            # Convert data to cartesian coordinates
            angle = []
            ran = []
            intensity = []
            myarr=np.empty((0,3), float)
            for point in scan.points:
                angle.append(point.angle);
                ran.append(point.range);
                intensity.append(point.intensity);
                myarr=np.append(myarr,[[point.angle*180/3.142,point.range,point.intensity]],axis=0)
                smallarr=myarr[67:202,:]

            # Update LIDAR plot
            self.lidar_polar.clear()
            self.lidar_polar.scatter(angle, ran, c=intensity, cmap='hsv', alpha=0.95)

            self.lidar_polar.set_rmax(6)
            self.lidar_polar.set_title('LIDAR Plot')
            self.canvas.draw()
            # Define input array
            input_arr = smallarr
            # print("check 2")

            # Get the second column of the input array
            input_col = input_arr[:, 1]

            # Interpolate the values if the input array is smaller than the new array
            if input_col.size < new_width:
                x_old = np.linspace(0, 1, input_col.size)
                x_new = np.linspace(0, 1, new_width)
                f = interp1d(x_old, input_col, kind='linear')
                input_col = f(x_new)

            # Calculate the center row of the new array
            center_row = int(new_arr.shape[0]/2)
            new_arr[center_row-20:center_row+20, :] = input_col
            depth_lidar= new_arr


        # Get current ZED camera image
        self.cam.grab()
        depth_image=sl.Mat()
        depth = sl.Mat(self.cam.get_camera_information().camera_resolution.width, self.cam.get_camera_information().camera_resolution.height, sl.MAT_TYPE.U16_C1)

        image = self.cam.retrieve_image(depth_image,sl.VIEW.DEPTH)
        depthvalue=self.cam.retrieve_measure(depth,sl.MEASURE.DEPTH)
        if image is not None:
            # Convert image to Tkinter-compatible format
            image = depth_image.get_data()
            #print(image.shape)
            depth_stereo=depth.get_data() # for fusion
            depthreal=depth_stereo.astype(np.uint8)  # for display in cv2 in color
            imagereal = depth_image.get_data() # for display in cv2 in grayscale

            # Fuse LIDAR and ZED camera depth maps
            fused_depth_map = np.zeros_like(depth_lidar)
            for i in range(depth_lidar.shape[0]):
              for j in range(depth_lidar.shape[1]):
                 fused_depth_map[i, j] = self.fuse_depth_values(depth_stereo[i, j], depth_lidar[i, j], x, P, A, B, C, Q, R)

            # Centred the text in the image
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = "Stereo Depth Map"
            textsize = cv2.getTextSize(text, font, 1, 2)[0]
            textX = (imagereal.shape[1] - textsize[0]) // 2
            textY = (textsize[1] // 2) + 5
            cv2.putText(imagereal, text, (textX, textY), font, 1, (255, 255, 255), 2)
           # normalized_image = cv2.normalize(depthreal, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
           # colored_image = cv2.applyColorMap(normalized_image, cv2.COLORMAP_JET)
            #img1 = Image.fromarray(colored_image)
          #  print(fused_depth_map.shape)
            fused_depth_map=fused_depth_map.astype(np.uint16)
            fused_depth_map_normalized = cv2.normalize(fused_depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
          #  fused_colored_image = cv2.applyColorMap(fused_depth_map_normalized, cv2.COLORMAP_JET)
          #  img1 = Image.fromarray(fused_colored_image)
            img1=Image.fromarray(fused_depth_map_normalized)

            img = Image.fromarray(imagereal)
            img_tk = ImageTk.PhotoImage(img)

            img_tk1 = ImageTk.PhotoImage(img1)

            # Update image label
            self.image_label.config(image=img_tk)
            self.image_label.image = img_tk

            self.image_label1.config(image=img_tk1)
            self.image_label1.image = img_tk1

        # Set timer for next update
        self.timer = self.after(100, self.update_lidar_and_image)

    def close(self):
        # Close YDLidar and ZED camera connections
        self.laser.turnOff()
        self.cam.close()

        # Close the application window
        self.destroy()

if __name__ == "__main__":
    app = LidarMainWindow()
    app.mainloop()

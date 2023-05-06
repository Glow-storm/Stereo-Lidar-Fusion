import cv2
import numpy as np

# Open the video file
video = cv2.VideoCapture("D:/dataspell project/FinalYearProject/FinalYearProject/2011_09_26_drive_0001_syncimage_02/depth maps generated/output.mp4")

frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))

# Create the Jet colormap
color_map = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv2.COLORMAP_JET)

# Create the legend image
legend = np.zeros((25 * (256//25), 25, 3), dtype=np.uint8)
for j in range(256//25):
    for i in range(25):
        pixel_value = j*25 + i
        if pixel_value <= 50:
            legend[j*25:j*25+25, i] = [255, 0, 0]  # Red
        elif pixel_value >= 200:
            legend[j*25:j*25+25, i] = [0, 0, 255]  # Blue
        else:
            legend[j*25:j*25+25, i] = color_map[pixel_value, 0]  # Jet colormap

# Add the corresponding pixel values to the legend
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.4
font_thickness = 1
for j in range(256//25):
    for i in range(25):
        pixel_value = j*25 + i
        position = (i * 25 + 5, (j+1) * 25 - 5)
        cv2.putText(legend, str(pixel_value), position, font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)

# Define the video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
output_video = cv2.VideoWriter("outputfinal.mp4", fourcc, 1, ( frame_width*2, frame_height))
while True:
    ret, frame = video.read()

    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Add the legend to the right of the video frame
    frame_height, frame_width, _ = frame.shape
    legend_height, legend_width, _ = legend.shape

    # Resize the legend to match the height of the frame
    legend_resized = cv2.resize(legend, (int(frame_height * legend_width / legend_height), frame_height))

    if frame_height != legend_height:
        max_height = max(frame_height, legend_height)
        frame_dif = max_height - frame_height
        legend_dif = max_height - legend_resized.shape[0] # use the resized legend
        frame = cv2.copyMakeBorder(frame, 0, frame_dif, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        legend_resized = cv2.copyMakeBorder(legend_resized, 0, legend_dif, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    output = np.hstack((frame, legend_resized))
    output_video.write(output)

    cv2.imshow("Video with Legend", output)
    key = cv2.waitKey(30)

# If the user pressed 'q', break out of the loop
    if key == ord('q'):
        break

# If the end of the video is reached, set the current frame position back to the beginning
    if video.get(cv2.CAP_PROP_POS_FRAMES) == video.get(cv2.CAP_PROP_FRAME_COUNT):
        video.set(cv2.CAP_PROP_POS_FRAMES, 0)
video.release()
cv2.destroyAllWindows()

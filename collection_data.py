import pyrealsense2 as rs
import numpy as np
import cv2
import os
import dlib
import pandas as pd
import time

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# collecting data
depth_points = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 'label']
data_points = [9, 34, 37, 40, 43, 46, 49, 52, 55, 58]

# Load the detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# time for collecting data
seconds = 0
past_seconds = seconds
count = 0

data = {
    "Chin": [],
    "Nose": [],
    "RightEye1": [],
    "RightEye2": [],
    "LeftEye1": [],
    "LeftEye2": [],
    "MouthRight": [],
    "MouthTop": [],
    "MouthLeft": [],
    "MouthBottom": [],
    "Label": []
}
df = pd.DataFrame(data)

# Start streaming
pipeline.start(config)

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Receiving time
        seconds = time.time()

        # Convert image into grayscale
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        # Use detector to find landmarks
        faces = detector(gray)
        for face in faces:
            x1 = face.left()  # left point
            y1 = face.top()  # top point
            x2 = face.right()  # right point
            y2 = face.bottom()  # bottom point

            # Create landmark object
            landmarks = predictor(gray, face)

            # Loop through all the points
            for n in data_points:
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                try:
                    cv2.circle(color_image, (x, y), 3, (0, 255, 0), -1)
                    depth_points[data_points.index(n)] = depth_image[y, x]
                except IndexError:
                    pass

        # Export data to DataFrame
        if seconds - past_seconds >= 5 and 0 not in depth_points:
            past_seconds = seconds

            #Label = 'Real_Face'
            Label = 'Photo'

            depth_points[10] = Label
            df.loc[count] = depth_points

            print("\033[1;35;40m ================================================================")
            print(depth_points)
            print("\033[1;36;40m Index: " + str(count) + 
                  "\033[1;32;40m <Successfully Save!>")
            count += 1

        cv2.putText(color_image, str(count), (100, 50),
                    cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255), 2)

        # Show images
        cv2.namedWindow('RealSense Depth', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense Depth', color_image)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            
            print("\033[1;33;40m ")
            print(df)
            #df.to_csv('Dataset/Dataset_Depth_Face3.csv')
            df.to_csv('Dataset/Dataset_Depth_Photo3.csv')

            break

except Exception as e:
    print("\033[1;31;40m----------------------------------")
    print(e)

finally:

    # Stop streaming
    pipeline.stop()

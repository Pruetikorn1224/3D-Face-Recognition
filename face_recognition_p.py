import pyrealsense2 as rs
import cv2

import numpy as np
import pandas as pd
import os
import time

import dlib
import face_recognition as facerec
import tensorflow as tf

#=============== Reference Link ===============
#
# Face Recognition
#http://www.daydev.com/developer/s6-programming-language/python/face-recognition-real-time-python-opencv.html
#
# Intel Realsense SDK
#https://github.com/IntelRealSense/librealsense
#
# Tensorflow: Binary classifier
#https://towardsdatascience.com/performing-classification-in-tensorflow-95368fde289c
#https://medium.com/analytics-vidhya/save-and-load-a-tensorflow-estimator-model-for-predictions-233b798620a9
#
#=============================================

# Detect data during VIDEO
data_locations = []
data_encodings = []
data_names = []
frameProcess = True
# Database for recognition
face_names = []
face_encodings = []

# Attribute for frame
colorF = (255, 0, 0)
colorL = (255, 255, 255)
colorT = (0, 0, 255)
font = cv2.FONT_HERSHEY_SIMPLEX

name = 'UNKNOWN'

# Import all face from ./Profile
# Name your picture as your name
# Such as "PRUETIKORN.jpg"
entries = os.listdir('Profile/')
for pic in entries:
    if '.jpg' in pic or '.png' in pic:
        face_names.append(pic.split('_')[0])
        database_image = facerec.load_image_file("Profile/"+pic)
        database_encoding = facerec.face_encodings(database_image)[0]
        face_encodings.append(database_encoding)

# dataFrame from depth image
depth_points = {
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
}
input_fn = pd.DataFrame(data = depth_points, dtype = np.float64)

# Collecting data from depth image
depth_ps = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
data_ps = [9, 34, 37, 40, 43, 46, 49, 52, 55, 58]

# Load the model
path_model = './Model_dir/model'
new_model = tf.saved_model.load(path_model)

# Load the detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

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

#predict the output from the trained model 
def predict(dfeval, importedModel):
    colNames = dfeval.columns
    dtypes = dfeval.dtypes
    predictions = []
    for row in dfeval.iterrows():
        example = tf.train.Example()
        for i in range(len(colNames)):
            dtype = dtypes[i]
            colName = colNames[i]
            value = row[1][colName]
            if dtype == "object":
                value = bytes(value, "utf-8")
                example.features.feature[colName].bytes_list.value.extend(
                    [value])
            elif dtype == "float":
                example.features.feature[colName].float_list.value.extend(
                    [value])
            elif dtype == "int":
                example.features.feature[colName].int64_list.value.extend(
                    [value])
   
        predictions.append(
           importedModel.signatures["predict"](
           examples=tf.constant([example.SerializeToString()])))
    return predictions

#optional features
max_time = 0
record = cv2.VideoWriter('presenter.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, (640, 480))

# Start streaming
pipeline.start(config)

try:
    while True:
        time_s = time.time()	

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        resizing = cv2.resize(color_image, (0, 0), fx=0.25, fy=0.25)
        result = resizing[:, :, ::-1]
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        #Check that there is a face or not
        if frameProcess:
            data_l = detector(gray)
            data_n = []
            newPred = []
            for f in data_l:
                landmarks = predictor(gray, f)
                for n in data_ps:
                     x = landmarks.part(n).x
                     y = landmarks.part(n).y
                     try:
                         depth_ps[data_ps.index(n)] = depth_image[y, x]
                     except IndexError:
                         pass
            input_fn.loc[0] = depth_ps
            face_pred = predict(input_fn, new_model)
            for pred in face_pred:
                newPred.append(np.argmax(pred["probabilities"]))

            data_l = facerec.face_locations(result)
            data_e = facerec.face_encodings(result, data_l)
            for enc in data_e:
                recog = facerec.compare_faces(face_encodings, enc)
                name = 'UNKNOWN'
                if True in recog:
                    match_index = recog.index(True)
                    name = face_names[match_index]
                    data_n.append(name)

        frameProcess = not frameProcess

        for (t, r, b, l), n, p in zip(data_l, data_n, newPred):
            if p == 1:
                t *= 4
                r *= 4
                b *= 4
                l *= 4
                cv2.rectangle(color_image, (l, t), (r, b), colorF, 2)
                cv2.rectangle(color_image, (l, b-35), (r, b), colorF, cv2.FILLED)
                cv2.putText(color_image, n, (l+6, b-6), font, 1.0, colorL, 1)

        # Show images
        record.write(color_image)
        cv2.namedWindow('RealSense Face Recognition', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense Face Recognition', color_image)
        key = cv2.waitKey(1)
	
        process_time = time.time() - time_s
        print('\033[1;35;40m Time process: ' + str(process_time) + ' ms')
        if max_time < process_time:
            max_time = process_time
        

        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break

finally:
    print("\033[1;36;40m-----------------------------------------------------------------")
    print('Maximum time: ' + str(max_time) + ' ms')

    # Stop streaming
    pipeline.stop()

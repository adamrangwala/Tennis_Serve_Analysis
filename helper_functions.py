
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
import mediapipe as mp
import numpy as np
import math
import mimetypes
import time
import sys
from tqdm import tqdm
import locale
locale.getpreferredencoding = lambda: "UTF-8"
import ultralytics
ultralytics.checks()
from ultralytics.utils import yaml_load
from ultralytics.utils.checks import check_yaml
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog


def promptUserForFile():
    root = tk.Tk()
    root.withdraw()  # Hide the main tkinter window
    # Ask the user to select a file
    file_path = filedialog.askopenfilename(
    title="Select a Video File",
    filetypes=[
              ("All Files", "*.*")      # Optional: Allow all file types
    ]
    )
    return file_path
'''
if file_path:
    print(f"Selected file: {file_path}")
else:
    print("No file selected.")
    file_path = filedialog.askopenfilename(
        title="Select a Serve Video Clip",
        initialdir="C:/Users",  # Default to Desktop (adjust as needed)
        filetypes=[("Video Files", "*.mp4*", "*.MOV"), ("Image Files", "*.jpeg*", "*.png")]
    )

    if file_path:
        print(f"You selected: {file_path}")
        return file_path
    else:
        print("No file selected.")
        '''
#guess filetype from input file
def determineInputFile(file_path):

    file_type, encoding = mimetypes.guess_type(file_path)
    file_type = file_type.split('/')
    file_type = file_type[1]
    if file_type in ["jpeg", "png"]:
        type_name = "image"
        print(type_name)
    elif file_type in ["quicktime", "mp4"]:
        type_name = "video"
        print(type_name)
    else:
        type_name = "other"
        print(type_name)
    return type_name

# Get Video Properties from Video Capture Object
def getVideoProperties(cap):
    frame_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frame_size = (int(frame_w), int(frame_h))
    fps = cap.get(cv2.CAP_PROP_FPS)
    return frame_w, frame_h, frame_size, fps

def setupVideoOutput(file_name, fps, frame_size, output_dir):
    prefix = "_Analyzed.mp4"
    out_name = file_name[:-4] + prefix
    print ("Outname", out_name)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

      # Debugging information
    print(f"File name: {file_name}")
    print(f"Output directory: {output_dir}")
    print(f"Full output path: {out_name}")
    print(f"FPS: {fps}, Frame Size: {frame_size}")

    vid_out = cv2.VideoWriter(out_name, fourcc, fps, frame_size)

    if not vid_out.isOpened():
        print("Error: VideoWriter failed to open.")
        return None
    return vid_out

def applyClahe(img):
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Split HSV channels
    h, s, v = cv2.split(hsv_image)
    # Apply CLAHE to the Value channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_v = clahe.apply(v)
    # Merge the enhanced V channel back with H and S
    enhanced_hsv = cv2.merge((h, s, enhanced_v))
    # Convert back to BGR color space
    clahe_img = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)
    #cv2.imshow("CLAHE Image", clahe_img) #debugging
    return clahe_img

def detect(frame, network):
    """Detects whether a given frame of contains (un)safe social distancing."""
    results = []
    h, w = frame.shape[:2]

    # Pre-processing: mean subtraction and scaling to match model's training set.
    blob = cv2.dnn.blobFromImage(
        frame, 0.007843, (300, 300), [127.5, 127.5, 127.5])
    network.setInput(blob)

    # Run an inference of the model, passing blob through the network.
    network_output = network.forward()

    # Loop over all results.
    for i in np.arange(0, network_output.shape[2]):
        class_id = network_output[0, 0, i, 1]
        confidence = network_output[0, 0, i, 2]

        # Filter for only detected people (classID 15) and high confidence.
        if confidence > 0.15 and class_id == 15:
            # Remap 0-1 position outputs to size of image for bounding box.
            box = network_output[0, 0, i, 3:7] * np.array([w, h, w, h])
            box = box.astype('int')
            
            box_height = box[3]-box[1]

            # Calculate the person center from the bounding box.
            center_x = int((box[0] + box[2]) / 2)
            center_y = int((box[1] + box[3]) / 2)

            results.append((confidence, box, (center_x, center_y), box_height))
    return results

def getScaledBbox(results, main_list_max_index, img_w, img_h, scaling_factor):
    """
    Calculates pt1 (top left) and pt 2 (bottom right) of scaled boundary square around detected human

    Paramaters:
    img(array): main img to pass in.
    results (human detection list): list of list of confidence, bbox, centroid and bbox height
    main_list_max_index (int): index of the results sublist with largest detected human
    scale_factor (float): factor to multiply bbox to ensure human pose falls within box (for cropping)
     
    Returns:
        2 Tuples: pt 1 - top left of scaled bbox, 
    """
    centroid_x, centroid_y = results[main_list_max_index][2]
    box_height = results[main_list_max_index][3]
    box_scaled_h = box_height * scaling_factor
    top_left_x = int(centroid_x - box_scaled_h/2)
    top_left_y = int(centroid_y - box_scaled_h/2)
    bottom_right_x = int(centroid_x + box_scaled_h/2)
    bottom_right_y = int(centroid_y + box_scaled_h/2)
    
    # If box falls outside of img dimensions, use img dimensions instead (effectively ignore)
    if top_left_x < 0:
        top_left_x = 0
    if top_left_y < 0:
        top_left_y = 0
    if bottom_right_x > img_w:
        bottom_right_x = int(img_w) 
    if bottom_right_y > img_h:
        bottom_right_y = int(img_h)
    
    # Retrieve scaled boundary box points.
    pt1 = (top_left_x, top_left_y)
    pt2 = (bottom_right_x, bottom_right_y)
    

    return pt1, pt2

def scaledCropImg(img, pt1, pt2, color=(0, 0, 255)):
    #cv2.rectangle(img, pt1, pt2, color, 1) # uncomment for debugging
    cropped_img = img[pt1[1]:pt2[1], pt1[0]:pt2[0]]
    #cv2.imshow("Cropped Image", cropped_img) # debugging
    return cropped_img

def resizeImg(img, bottom_right_pt, min_res = 450):

    desired_width = min_res
    img_h, img_w = img.shape[:2]
    aspect_ratio = img_w/img_h
    desired_height = int(desired_width * aspect_ratio)
    dim = (desired_width, desired_height)
    #print("resized Dim: ", dim)
    if bottom_right_pt[0] < min_res or bottom_right_pt[1] < min_res:
        resized_img = cv2.resize(img, dsize = dim, interpolation = cv2.INTER_AREA)
    else:    
        resized_img = img
    #cv2.imshow("resized_img", resized_img) #debugging

    return resized_img

def estimPose_img(input_file, pose, mp_pose):
    
    # Convert the image from BGR into RGB format.
    RGB_img = cv2.cvtColor(input_file, cv2.COLOR_BGR2RGB)
    # Perform the Pose Detection.
    landmarks = pose.process(RGB_img)
    # Retrieve the height and width of the input image.
    height, width = input_file.shape[:2]
    # Initialize a list to store the detected landmarks.
    landmarks_list = []
    # Check if any landmarks are detected.
    if landmarks.pose_landmarks:
        # Iterate over the detected landmarks.
        for landmark in landmarks.pose_landmarks.landmark:
            landmarks_list.append((int(landmark.x * width), int(landmark.y * height),
                                  (landmark.z)))
    landmarks = landmarks.pose_landmarks
    return landmarks_list, landmarks
    
def getTennisLandmarks(landmarks, enum_pose):
    """
    Retrive the specific landmarks selected my coder for tennis applications.
    Specifically, Both right and left components of:
    Nose, Shoulder, Elbow, Wrist, Hip, Knee, Ankle, Heel, Foot

    Paramaters:
    landmarks(landmark_list): All 33 landmarks with REAL coordinates of shape (33,3)
    enum_pose: readability list-like object that maps a body part with an index
    
    Returns:
        List: (x,y,z) coordinates of each selected body part
    """
    tennis_landmarks = []
    #tennis_landmarks.append(landmarks[enum_pose.NOSE])
    tennis_landmarks.append(landmarks[enum_pose.RIGHT_SHOULDER])
    tennis_landmarks.append(landmarks[enum_pose.LEFT_SHOULDER])
    tennis_landmarks.append(landmarks[enum_pose.RIGHT_ELBOW])
    tennis_landmarks.append(landmarks[enum_pose.LEFT_ELBOW])
    tennis_landmarks.append(landmarks[enum_pose.RIGHT_WRIST])
    tennis_landmarks.append(landmarks[enum_pose.LEFT_WRIST])
    tennis_landmarks.append(landmarks[enum_pose.RIGHT_HIP])
    tennis_landmarks.append(landmarks[enum_pose.LEFT_HIP])
    tennis_landmarks.append(landmarks[enum_pose.RIGHT_KNEE])
    tennis_landmarks.append(landmarks[enum_pose.LEFT_KNEE])
    tennis_landmarks.append(landmarks[enum_pose.RIGHT_ANKLE])
    tennis_landmarks.append(landmarks[enum_pose.LEFT_ANKLE])
    tennis_landmarks.append(landmarks[enum_pose.RIGHT_HEEL])
    tennis_landmarks.append(landmarks[enum_pose.LEFT_HEEL])
    tennis_landmarks.append(landmarks[enum_pose.RIGHT_FOOT_INDEX])
    tennis_landmarks.append(landmarks[enum_pose.LEFT_FOOT_INDEX])


    return tennis_landmarks

def displayBackKneeFlexion(output_img, landmarks, enum_pose, frame_w, frame_h, color_light = (255,255,0), color_join = (0,20,200)):
    
        r_hip_p = get_landmark_point(landmarks, enum_pose.RIGHT_HIP, frame_w, frame_h)
        r_knee_p = get_landmark_point(landmarks, enum_pose.RIGHT_KNEE, frame_w, frame_h)
        r_heel_p = get_landmark_point(landmarks, enum_pose.RIGHT_HEEL, frame_w, frame_h)
       
        # Circle Landmarks of interest.
        cv2.circle(output_img, (r_hip_p[0], r_hip_p[1]), 3, color_light, -1)
        cv2.circle(output_img, (r_knee_p[0], r_knee_p[1]), 3, color_light, -1)
        cv2.circle(output_img, (r_heel_p[0], r_heel_p[1]), 3, color_light, -1)
        
        # Compute angle between hip - knee - heel
        v_knee_hip = np.subtract(r_hip_p, r_knee_p)
        v_knee_heel = np.subtract(r_knee_p, r_heel_p)
        angle_hip_knee_heel = 180 - compute_angle(v_knee_heel, v_knee_hip)
        text_loc = (r_knee_p[0] - 75, r_knee_p[1] - 5)
        cv2.putText(output_img, "back: " + str(int(angle_hip_knee_heel)), text_loc, cv2.FONT_HERSHEY_SIMPLEX, .5, 
                    color_join, 1, cv2.LINE_AA)
        
        # Draw angle lines on img
        cv2.line(output_img, (r_hip_p[0], r_hip_p[1] ), (r_knee_p[0], r_knee_p[1]), color_join, 2, cv2.LINE_AA)
        cv2.line(output_img, (r_knee_p[0], r_knee_p[1] ), (r_heel_p[0], r_heel_p[1]), color_join, 2, cv2.LINE_AA)

def displayFrontKneeFlexion(output_img, landmarks, enum_pose, frame_w, frame_h, color_light = (255,255,0), color_join = (0,20,200)):
    
        l_hip_p = get_landmark_point(landmarks, enum_pose.LEFT_HIP, frame_w, frame_h)
        l_knee_p = get_landmark_point(landmarks, enum_pose.LEFT_KNEE, frame_w, frame_h)
        l_heel_p = get_landmark_point(landmarks, enum_pose.LEFT_HEEL, frame_w, frame_h)
       
        # Circle Landmarks of interest.
        cv2.circle(output_img, (l_hip_p[0], l_hip_p[1]), 3, color_light, -1)
        cv2.circle(output_img, (l_knee_p[0], l_knee_p[1]), 3, color_light, -1)
        cv2.circle(output_img, (l_heel_p[0], l_heel_p[1]), 3, color_light, -1)
        
        # Compute angle between hip - knee - heel
        v_knee_hip = np.subtract(l_hip_p, l_knee_p)
        v_knee_heel = np.subtract(l_knee_p, l_heel_p)
        angle_hip_knee_heel = 180 - compute_angle(v_knee_heel, v_knee_hip)
        text_loc = (l_knee_p[0] + 45, l_knee_p[1] - 15)
        cv2.putText(output_img, "front: " + str(int(angle_hip_knee_heel)), text_loc, cv2.FONT_HERSHEY_SIMPLEX, .5, 
                    color_join, 1, cv2.LINE_AA)
        
        # Draw angle lines on img
        cv2.line(output_img, (l_hip_p[0], l_hip_p[1] ), (l_knee_p[0], l_knee_p[1]), color_join, 2, cv2.LINE_AA)
        cv2.line(output_img, (l_knee_p[0], l_knee_p[1] ), (l_heel_p[0], l_heel_p[1]), color_join, 2, cv2.LINE_AA)

def compute_angle(v1, v2):

    # Unit vector.
    v1u = v1 / np.linalg.norm(v1)
    # Unit vector.
    v2u = v2 / np.linalg.norm(v2)
    # Compute the angle between the two unit vectors.
    angle_deg = np.arccos(np.dot(v1u, v2u)) * 180 / math.pi

    return angle_deg
def get_landmark_point(landmarks, landmark_point, w, h):
    x = int(landmarks.landmark[landmark_point].x * w)
    y = int(landmarks.landmark[landmark_point].y * h)
    point = np.array([x, y])
    return point

def getBodyPartCoordinate(landmarks, body_part, frame_w, frame_h):
    """
    Retrieve the scaled (x, y) coordinates of a specific body part.
    
    Parameters:
        landmarks (landmark_list): The landmarks from MediaPipe pose detection.
        body_part (mp_pose.PoseLandmark): The body part enumeration to retrieve.
        frame_w (int): Width of the frame.
        frame_h (int): Height of the frame.
    
    Returns:
        np.array: The (x, y) coordinates of the specified body part.
    """
    x = int(landmarks.landmark[body_part].x * frame_w)
    y = int(landmarks.landmark[body_part].y * frame_h)
    return np.array([x, y])
    
def setTrackerType(tracker_type):
    if tracker_type == 'CSRT':
        tracker = cv2.legacy.TrackerCSRT_create()
    elif tracker_type == 'KCF':
        tracker = cv2.legacy.TrackerKCF_create()
    elif tracker_type == 'MEDIANFLOW':
        tracker = cv2.legacy.TrackerMedianFlow_create()  
    elif tracker_type == 'MIL':
        tracker = cv2.legacy.TrackerMIL_create()
    elif tracker_type == 'MOSSE':
        tracker = cv2.legacy.TrackerMOSSE_create()
    elif tracker_type == 'TLD':
        tracker = cv2.legacy.TrackerTLD_create()
    else:
        print('invalid tracker type')
        return 
    return tracker

# Prompt User to Click and Drag Over Ball Location
def getStartingBallLocation(frame):
    bbox_x, bbox_y, bbox_w, bbox_h = cv2.selectROI('Select ROI', frame, showCrosshair=True, fromCenter=True)
    bbox = [bbox_x, bbox_y, bbox_w, bbox_h]
    return bbox, bbox_x, bbox_y, bbox_w, bbox_h

# Define Annotation Convenience Functions.
def drawBannerText(frame, text, banner_height_percent=0.08, font_scale=.8, text_color=(0, 255, 0), 
                   font_thickness=2):
    # Draw a black filled banner across the top of the image frame.
    # percent: set the banner height as a percentage of the frame height.
    banner_height = int(banner_height_percent * frame.shape[0])
    cv2.rectangle(frame, (0, 0), (frame.shape[1], banner_height), (0, 0, 0), thickness=-1)

    # Draw text on banner.
    left_offset = 20
    location = (left_offset, int(15 + (banner_height_percent * frame.shape[0]) / 2))
    cv2.putText(frame, text, location, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 
                font_thickness, cv2.LINE_AA)

def drawRectangle(frame, bbox, color=(255,0,0)):
    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    cv2.rectangle(frame, p1, p2, color, 2, 1)

def displayRectangle(frame, bbox, color=(255,0,0)):
    frameCopy = frame.copy()
    drawRectangle(frameCopy, bbox, color)
    cv2.putText(frameCopy, 'Press any key to continue', (20,40), cv2.FONT_HERSHEY_SIMPLEX, 
        1, (0,255,0), 1, cv2.LINE_AA)
    #cv2.imshow('Rectangle', cv2.resize(frameCopy, None, fx=0.5, fy=0.5))
    cv2.waitKey(0)
    cv2.destroyAllWindows() 
     
def drawText(frame, text, location=(20,20), font_scale=1, color=(50,170,50), font_thickness=2):
    cv2.putText(frame, text, location, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 
                font_thickness, cv2.LINE_AA)

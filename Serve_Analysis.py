from helper_functions import *

#----- SETTING UP INPUTS & OUTPUTS -----# 
output_dir = r"C:\Users\rangw\Dropbox\Open_CV\Mastering_OpenCV_with_Python\Adam_Mini_Projects\Serve_Analysis"
print("Output Directory: ", output_dir)
# Select image or video file
file_name = promptUserForFile()
print("File name: ", file_name)

# Determine Input file type
file_type = determineInputFile(file_name) # mp4, quicktime -> vid | jpeg, png -> photo

# Read input video file 
if file_type == "video": 
    cap = cv2.VideoCapture(file_name)
    # Check for empty video
    if not cap.isOpened():
        print("Error: Cannot open video source")
    # Get Input Video Properties    
    frame_w, frame_h, frame_size, fps = getVideoProperties(cap)

# Read input image file
if file_type == "image":
    img = cv2.imread(file_name)
    output_name = "ANALYZED_" + file_name
    # Check if image is loaded
    if img is None:
        print("Error: Unable to load image.")
    else:
        img_w = img.shape[1]
        img_h = img.shape[0]

# DNN Object(Human) Detection Initialization
# Load in the pre-trained SSD model.
configFile = 'C:\\Users\\rangw\\Dropbox\\Open_CV\Mastering_OpenCV_with_Python\\Adam_Mini_Projects\\Serve_Analysis\\MobileNetSSD_deploy.prototxt'
modelFile = 'C:\\Users\\rangw\\Dropbox\\Open_CV\Mastering_OpenCV_with_Python\\Adam_Mini_Projects\\Serve_Analysis\\MobileNetSSD_deploy.caffemodel'
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

#---- POSE ESTIMATION ----#

# Pose Init
mp_pose = mp.solutions.pose


# BGR Colors
color_light  = (255, 255, 0)
color_marks  = (0, 255, 255)
color_join   = (0, 20, 200)
enum_pose  = mp_pose.PoseLandmark

### Landmark Estimation for Image
if file_type == "image":
    print("Input file is image") # debugging tool
    ###--- PREPROCESSING ---###            
    cv2.imshow("original img", img)

    # CLAHE - low-light local contract with minimal noise amplification
    img = applyClahe(img)

    # Brighten img
    matrix = np.ones(img.shape, dtype = 'uint8') * 1
    # Create brighter and darker images.
    img_brighter = np.clip(cv2.add(img, matrix),0,255)

    # Detect Bounding Box for Human using DNN
    results = detect(img_brighter, network=net) 
    
    # Specify index of box_h that we want to get max height of
    sublist_index = 3

    # Get index of main list that contains the max value at specified sublist index
    main_list_max_index = max(range(len(results)), key = lambda i: results[i][sublist_index])
    print("Person detected with ", int(results[main_list_max_index][0]*100), "'%' confidence") 
          
    # Get a scaled boundary square top left and bottom right pt for cropping largest human
    top_left_pt, bottom_right_pt = getScaledBbox(results, main_list_max_index, img_w, img_h, scaling_factor=1.5)
    
    # Preprocessing: Scaled crop of the biggest detected human
    cropped_img = scaledCropImg(img_brighter, top_left_pt, bottom_right_pt)

    # Rescale Image for pre-processing before human pose est if too small (below 400 x 450)
    resized_img = resizeImg(cropped_img, bottom_right_pt)
   
    output_img = resized_img

    # Initializing mediapipe pose class.
    mp_pose = mp.solutions.pose

    # Setting up the Pose model for images.
    pose_img = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, model_complexity=2)
    # Initializing mediapipe drawing class to draw landmarks on specified image.
    mp_drawing = mp.solutions.drawing_utils

    # creates list of landmarks with x,y,z real coordinates
    landmarks_list, landmarks = estimPose_img(output_img, pose=pose_img, mp_pose=mp_pose)

    '''tennis_landmarks = getTennisLandmarks(landmarks, enum_pose)
    if landmarks is not None:
        for body_part in tennis_landmarks:
            cv2.circle(output_img, (body_part[0], body_part[1]), 3, color_light, -1) ''' 
    #   Annotate key angles given a stage
    frame_w = output_img.shape[1]
    frame_h = output_img.shape[0]
    print("Initial Frame Size: ", frame_h, frame_w)
    if landmarks is not None:
        #---- Loading Stage ----#
        # Front Knee Flexion - right hip, right knee, right heel IF right-handed
        displayBackKneeFlexion(output_img, landmarks, enum_pose, frame_w, frame_h)
        displayFrontKneeFlexion(output_img, landmarks, enum_pose, frame_w, frame_h)

        
    else:
        print("could not retrieve tennis landmarks")
       
 
    # Show stages
    cv2.imshow("Output", output_img)
    

    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()

###---- IF INPUT IS VIDEO ----###.

if file_type == "video":
    
    # Initializing mediapipe pose class.
    mp_pose = mp.solutions.pose
    
    # Setting up the Pose model for Video.
    static_image_mode = False # since video
    model_complexity = 2 # tested better for tennis
    smooth_landmarks = True
    enable_segmentation = True # background segmentation
    min_detection_confidence = 0.3
    min_tracking_confidence = 0.3
    
    # Create Pose object with defined parameters
    pose_frame = mp_pose.Pose(static_image_mode, model_complexity, smooth_landmarks, enable_segmentation, min_detection_confidence, min_tracking_confidence)
    
    print('Input file is video')
    print('Processing video, please wait ...')
    
    # Initialize identifier for first frame
    first_frame = True

    # PROCESSING LOOP
    ###---- PREPROCESSING ----###
    while cap.isOpened():

        # Capture image to 'frame' variable
        has_frame, frame = cap.read()
        
        # If no frame left, end loop
        if not has_frame:
            print("No frame detected")
            break 

        # Start timer for fps calculation
        timer = cv2.getTickCount()

        # CLAHE - low-light local contract with minimal noise amplification
        frame = applyClahe(frame)

        # Brighten img
        matrix = np.ones(frame.shape, dtype = 'uint8') * 1
        # Create brighter and darker images.
        frame_brighter = np.clip(cv2.add(frame, matrix),0,255)

        if first_frame:
            # Detect Bounding Box for Human using DNN
            results = detect(frame_brighter, network=net) 
        
        if results == []:
            print("no human detected ")
            continue
        else:
            if first_frame:
                print ("results: ", results)
                # Specify index of box_h that we want to get max height of
                sublist_index = 3
                
                # Get index of main list that contains the max value at specified sublist index
                main_list_max_index = max(range(len(results)), key = lambda i: results[i][sublist_index])
                
                print("Person detected with ", int(results[main_list_max_index][0]*100), "'%' confidence") 
            
                # Get a scaled boundary square top left and bottom right pt for cropping largest human
                top_left_pt, bottom_right_pt = getScaledBbox(results, main_list_max_index, frame_w, frame_h, scaling_factor=2)
                print("Bounding Box: ", top_left_pt, bottom_right_pt)
        # Preprocessing: Scaled crop of the biggest detected human
        cropped_frame = scaledCropImg(frame_brighter, top_left_pt, bottom_right_pt)

        # Rescale Image for pre-processing before human pose est if too small (below 400 x 450)
        resized_frame = resizeImg(cropped_frame, bottom_right_pt)

        if first_frame:
                frame_size = resized_frame.shape[:2]
                vid_out = setupVideoOutput(file_name, fps, frame_size, output_dir)
                first_frame = False 

        if not vid_out.isOpened():
            print("Error: VideoWriter not initialized. Check codec, file path, and frame size.")
                
        output_frame = resized_frame
        # creates list of landmarks with x,y,z real coordinates
        landmarks_list, landmarks = estimPose_img(output_frame, pose=pose_frame, mp_pose=mp_pose)

        if landmarks is None:
                print("Warning: No landmarks detected in the current frame.")
                continue
        else: 
            print("landmarks detected: ", len(landmarks_list))
        
        #   Annotate key angles given a stage
        frame_w = output_frame.shape[1]
        frame_h = output_frame.shape[0]
        #---- Loading Stage ----#
        # Front Knee Flexion - right hip, right knee, right heel IF right-handed
        displayBackKneeFlexion(output_frame, landmarks, enum_pose, frame_w, frame_h)
        displayFrontKneeFlexion(output_frame, landmarks, enum_pose, frame_w, frame_h)

        # Calculate Frames per second (FPS)
        fps_banner = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        # Display Info
        drawBannerText(output_frame, 'FPS : ' + str(int(fps_banner)))
        output_frame = cv2.resize(output_frame, (frame_size))
        # Write frame to video
        vid_out.write(output_frame)
    
    
    cap.release()
    if vid_out.isOpened():
        vid_out.release()
        print("Video written successfully.")
    else:
        print("Failed to write video.")
        
    cv2.destroyAllWindows()
    print('Processing completed.')
   
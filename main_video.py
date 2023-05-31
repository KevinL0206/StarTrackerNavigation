import torch
import numpy as np
import cv2
from models.experimental import attempt_load
from utils.general import non_max_suppression
from utils.torch_utils import select_device
import math
from PIL import Image, ImageDraw
import random
import matplotlib.pyplot as plt
import glob
from mayavi import mlab
import plotly.graph_objs as go
import os
import torch.nn.functional as F
import pickle



os.chdir("c:/Users/klam0/Documents/University/code")
#42
def create_3d_line(start, end, color, width):
    line = go.Scatter3d(x=[start[0], end[0]], y=[start[1], end[1]], z=[start[2], end[2]],
                        mode='lines', line=dict(color=color, width=width), hoverinfo='skip')
    return line


def reverse_star_transform(new_stars, img_width, img_height, elevation, angle):
    original_stars = []
    
    for new_star in new_stars:
        new_x, new_y = new_star
        r = math.sqrt((new_x - img_width/2)**2 + (new_y - img_height/2)**2)
        theta = math.atan2(new_y - img_height/2, new_x - img_width/2) - math.radians(angle)
        
        x = r * math.cos(theta)
        y = r * math.sin(theta) / elevation
        
        star_x = x + img_width/2
        star_y = y + img_height/2

        # Calculate the center of the image
        center_x = int(img_width/2)
        center_y = int(img_height/2)

        # Calculate the distance between the center and each coordinate
        distances = (star_x - center_x, star_y - center_y)
        
        # Mirror the distances across the y-axis
        mirrored_distances = (-distances[0], distances[1])
        

        star_x,star_y = (center_x + mirrored_distances[0], center_y + mirrored_distances[1]) 
        original_stars.append((star_x, star_y))
    
    return original_stars

#-------------------------------------------------------------------------------------------------------------------------------------------------------#

#--Camera Calibration--#    
            
# Chessboard dimensions (number of internal corners)
chessboard_rows = 9
chessboard_cols = 13

# Prepare object points (3D points in the real world space)
objp = np.zeros((chessboard_rows * chessboard_cols, 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_cols, 0:chessboard_rows].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images
objpoints = []  # 3D points in the real world space
imgpoints = []  # 2D points in the image plane

# Load and process the chessboard images
#image_files = glob.glob("./*.jpg")  # Update the file path accordingly

scale_factor = 0.5
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
"""
for fname in image_files:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize the image
    width = int(gray.shape[1] * scale_factor)
    height = int(gray.shape[0] * scale_factor)
    dim = (width, height)
    resized_gray = cv2.resize(gray, dim, interpolation=cv2.INTER_AREA)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(resized_gray, (chessboard_cols, chessboard_rows), None)

    # If corners are found, refine them and add object points and image points
    if ret:
        refined_corners = cv2.cornerSubPix(resized_gray, corners, (11, 11), (-1, -1), criteria)
        objpoints.append(objp)
        imgpoints.append(refined_corners * (1 / scale_factor))

        
        # Visualize the detected corners for validation
        img = cv2.drawChessboardCorners(img, (chessboard_cols, chessboard_rows), refined_corners * (1 / scale_factor), ret)
        scaled_img = cv2.resize(img, (1000,1000), interpolation=cv2.INTER_AREA)
        cv2.imshow('Corners', scaled_img)
        cv2.waitKey(0)
        
        

#cv2.destroyAllWindows()

# Calibrate the camera
ret, camera_matrix, dist, rvecs1, tvecs1 = cv2.calibrateCamera(objpoints, imgpoints, resized_gray.shape[::-1], None, None)

# Save the variables to a file
with open('calibration_data.pkl', 'wb') as f:
    pickle.dump((ret, camera_matrix, dist, rvecs1, tvecs1), f)




print("Camera Calibration Completed")
"""
with open('calibration_data.pkl', 'rb') as f:
    ret, camera_matrix, dist, rvecs1, tvecs1 = pickle.load(f)
#-------------------------------------------------------------------------------------------------------------------------------------------------------#

# Load the YOLOv5 model
#44
device = select_device('CPU')
model = attempt_load('yolov5/runs/train/exp49/weights/best.pt', device=device)

# Open the input video file 26,
cap = cv2.VideoCapture('35n.mov')
width2 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height2 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the kernel for morphological opening
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))

URSA = [[433.8, 212.6],[433.8, 255.9],[354.9, 212.6],[340.5, 255.9],[298.1, 133.1],[270.1, 85.1],[209.4, 79.9]]

history = []
history2 = []
image_coordinates = []

# Loop through the frames of the video
while True:
    # Read a frame from the video
    ret, frame = cap.read()
    #print("Dimensions of frame:", frame.shape)
    # Stop the loop if we have reached the end of the video
    if not ret:
        break

    #frame = cv2.flip(frame, 1)

    # Convert the blurred frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform contrast stretching
    p1, p99 = np.percentile(gray, (1, 99))
    stretched_gray = np.uint8(np.clip((gray - p1) / (p99 - p1) * 255.0, 0, 255))

    # Apply Gaussian blur to the frame
    blurred_frame = cv2.GaussianBlur(gray, (9, 9), 0)

    # Apply a threshold to extract the bright pixels
    thresh = cv2.threshold(blurred_frame, 250, 255, cv2.THRESH_BINARY)[1]

    # Perform morphological opening to remove small bright regions
    opened_thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    #print("Dimensions of thresh:", opened_thresh.shape)
    # Create a named window with a custom size
    #cv2.namedWindow('thresh', cv2.WINDOW_NORMAL)
    #cv2.resizeWindow('thresh', 1000, 900)
    #cv2.imshow('thresh', opened_thresh)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(opened_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = 0 # Set a minimum area threshold
    max_area = 100000 # Set a maximum area threshold

    # Create an array to store the 2D positions of the LEDs
    led_positions = []

    filtered_contours = []

    # Draw bounding boxes around the detected contours and extract the centroid coordinates
    for contour in contours:
        
        #Limit size of contours found
        area = cv2.contourArea(contour)
        if min_area <= area <= max_area:
            filtered_contours.append(contour)    

            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate the centroid of the contour
            centroid_x = (x + w/2) 
            centroid_y = (y + h/2)
            # Add the centroid coordinates to the array of LED positions
            led_positions.append((centroid_x, centroid_y))

    height, width = opened_thresh.shape

    img_orig = Image.new('RGB', (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(img_orig)
    for star_num, star in enumerate(led_positions):
        draw.ellipse((star[0]-10, star[1]-10, star[0]+10, star[1]+10), fill='white')

    # Convert the frame to a PyTorch tensor
    img = np.asarray(img_orig)
   
    height, width = 512,512
    #height, width = opened_thresh.shape[:2]
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)

    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)

    # Convert the tensor to a NumPy array
    img_array = img_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    # Scale the pixel values to 0-255 and convert to uint8
    img_array = (img_array * 255).astype(np.uint8)
    
    # Display the image using OpenCV
    cv2.imshow('Image', img_array)

    # Make predictions using the YOLOv5 model
    with torch.no_grad():
        pred = model(img_tensor)[0]
    
    # Apply non-maximum suppression to remove overlapping detectionas
    pred = non_max_suppression(pred, conf_thres=0.4, iou_thres=0.6, classes=None, agnostic=False)
    print(pred)
    print(img_tensor.shape)

    # Extract bounding box coordinates and class labels from the output
    if len(pred) > 0:
        det = pred[0].cpu().numpy()
        bboxes = []
        labels = []
        confidences = []
        for *xyxy, conf, cls in det:
            x1, y1, x2, y2 = map(int, xyxy)
            x1 = int(x1 * frame.shape[1] / img.shape[1])
            y1 = int(y1 * frame.shape[0] / img.shape[0])
            x2 = int(x2 * frame.shape[1] / img.shape[1])
            y2 = int(y2 * frame.shape[0] / img.shape[0])
            bboxes.append([x1, y1, x2, y2])
            label = model.names[int(cls)]
            labels.append(label)
            confidences.append(conf)
        ratio_x = width 
        ratio_y = height 
        
        #print(bboxes)
        #print(labels)
        #print(confidences)

        # Draw bounding boxes and class labels on the frame
        for bbox, label in zip(bboxes, labels):
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


        if len(bboxes) > 0:
            for i, bbox in enumerate(bboxes):
                # Extract ROI from the original image using bounding box coordinates
                roi = opened_thresh[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            

            #print("Dimensions of new_img:", roi.shape)
            
            s = labels[i]
            parts = s.split('_')
            constellation = parts[0]
            angle = float(parts[1])
            elevation = float(parts[2])
            print(s)
        
            # Find contours in the thresholded image
            contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            min_area = 0 # Set a minimum area threshold
            max_area = 10000 # Set a maximum area threshold

            # Create an array to store the 2D positions of the LEDs
            led_positions = []

            filtered_contours = []

            # Draw bounding boxes around the detected contours and extract the centroid coordinates
            for contour in contours:
                
                #Limit size of contours found
                area = cv2.contourArea(contour)
                if min_area <= area <= max_area:
                    filtered_contours.append(contour)    

                    x, y, w, h = cv2.boundingRect(contour)
                    
                    

                    # Calculate the centroid of the contour
                    centroid_x = (x + w/2) + bbox[0]
                    centroid_y = (y + h/2) + bbox[1]


                    # Add the centroid coordinates to the array of LED positions
                    led_positions.append((centroid_x, centroid_y))

            print("initial",led_positions)
            # Get the dimensions of the image

            # Get the dimensions of the image
            height, width = frame.shape[:2]

            # Calculate the center of the image
            center_x = int(width / 2)
            center_y = int(height / 2)

            # Calculate the distance between the center and each coordinate
            distances = [(coord[0] - center_x, coord[1] - center_y) for coord in led_positions]
            
            # Mirror the distances across the y-axis
            mirrored_distances = [(-dist[0], dist[1]) for dist in distances]
            
            # Calculate the mirrored coordinates by adding the mirrored distances to the center
            led_positions = [(center_x + dist[0], center_y + dist[1]) for dist in mirrored_distances]
            print("mirror",led_positions)

            #print("LED Position:", led_positions)
            
            
            reverse_positions = reverse_star_transform(led_positions, width2, height2, elevation, 360 - angle)

            print("Reverse Position:", reverse_positions)
            
            paired_list = list(zip(led_positions, reverse_positions))
            # Sort the paired list based on the x-component (0th index) of the second element of each pair (from list2)
            sorted_pairs = sorted(paired_list, key=lambda pair: pair[1][0])

            # Extract the first elements (from list1) of the sorted pairs
            image_coordinates = [pair[0] for pair in sorted_pairs]
            sorted_list2 = [pair[1] for pair in sorted_pairs]
            print(image_coordinates)

            #print("coordinates",image_coordinates)
        #print("Sorted Reverse Position:", sorted_list2)
        #print("Real Position:", image_coordinates)


        """
        # Create new image object
        img_rot = Image.new('RGB', (width2, height2), (0, 0, 0))

        # Draw stars at new positions
        draw_rot = ImageDraw.Draw(img_rot)
        for x, star in enumerate(reverse_positions):
            size = 10 # generate a random size between 5 and 20
            draw_rot.ellipse((star[0]-size, star[1]-size, star[0]+size, star[1]+size), fill='white')
        
        img_rot_resized = img_rot.resize((1000, 900))

        img_rot_resized.show()
        """

        #-------------------------------------------------------------------------------------------------------------------------------------------------------#
        #--Distance Calculation--#

        if len(image_coordinates) > 3:

            points_2d = np.array(image_coordinates, dtype=np.float64)
            #print("2d", points_2d)
            #print("led",sorted(led_positions, key=lambda x: x[0]))
            numberofpoint = len(points_2d)
            print("number of point", numberofpoint)
            
            #points_3d = np.array([[28,17,-2.5],[28,20.5,-3],[22.5,17,-3.5],[21.5,20.5,-4],[19.3,13,-3.5],[17,8.5,-3.5],[12.5,8,-3.5]], dtype=np.float64)
            points_3d = np.array([[34.5, 12.5, -3.5], [38.5, 12.5, -4], [40.5, 17, -4.5], [44, 24, -2], [44.5, 20, -3.5], [48, 24, -2.5], [49, 20, -3.5]], dtype=np.float64)

            if len(points_2d) == len(points_3d):
                
                #print("2d",points_2d)
                #print("3d",points_3d)
                # Using solvePnP for camera pose estimation
                _, rvec, tvec = cv2.solvePnP(points_3d[0:numberofpoint], points_2d, camera_matrix, dist, flags=cv2.SOLVEPNP_SQPNP)

                # Refine the camera pose estimation using non-linear optimization
                rvec, tvec = cv2.solvePnPRefineLM(points_3d[0:numberofpoint], points_2d, camera_matrix, dist, rvec, tvec)
                
                # Calculate the rotation matrix from the rotation vector
                rotation_matrix, _ = cv2.Rodrigues(rvec) 
                
                rotation_matrix = abs(rotation_matrix) 
                print("Rotation Matrix:", rotation_matrix)

                # convert rotation vector to Euler angles
                theta_x, theta_y, theta_z = rvec.reshape(-1)

                print(f"Euler angles: ({theta_x:.2f}, {theta_y:.2f}, {theta_z:.2f})")

                # Calculate the camera's real-world coordinates
                camera_position = -np.matmul(rotation_matrix.T, tvec)
                camera_coordinate = camera_position.ravel()
                camera_coordinate[0] = abs(camera_coordinate[0])
                camera_coordinate[1] = abs(camera_coordinate[1])
                camera_coordinate[2] = -abs(camera_coordinate[2]) 
                history.append(camera_coordinate)

                print("Estimated Camera position (X, Y, Z):", camera_coordinate)
                print("History:", len(history))

                if len(history)== 100:

                    #real coordinates of LEDs
                    blue_coords = [[34.5,12.5,-3.5],[38.5,12.5,-4],[40.5,17,-4.5],[44,20,-3.5],[44,24,-2],[48,20,-3.5],[48,24,-2.5]]
                    red_coords = history

                    blue_x = [coord[0] for coord in blue_coords]
                    blue_y = [coord[1] for coord in blue_coords]
                    blue_z = [coord[2] for coord in blue_coords]

                    red_x = [coord[0] for coord in red_coords]
                    red_y = [coord[1] for coord in red_coords]
                    red_z = [coord[2] for coord in red_coords]

                    #Coordinates of the camera
                    fig = go.Figure()
                    fig.add_trace(go.Scatter3d(x=blue_x, y=blue_y, z=blue_z, mode='markers', marker=dict(color='blue', size=5)))
                    fig.add_trace(go.Scatter3d(x=red_x, y=red_y, z=red_z, mode='markers', marker=dict(color='red', size=5)))
                    #fig.add_trace(go.Scatter3d(x=[26], y=[53], z=[-10], mode='markers', marker=dict(color='green', size=5)))
                    #fig.add_trace(go.Scatter3d(x=[33.5], y=[53], z=[-20], mode='markers', marker=dict(color='green', size=5))) #26
                    #fig.add_trace(go.Scatter3d(x=[35], y=[15], z=[-50], mode='markers', marker=dict(color='green', size=5))) #28
                    #fig.add_trace(go.Scatter3d(x=[26], y=[15], z=[-50], mode='markers', marker=dict(color='green', size=5))) #31
                    #fig.add_trace(go.Scatter3d(x=[36], y=[15], z=[-50], mode='markers', marker=dict(color='yellow', size=5))) #31
                    #fig.add_trace(go.Scatter3d(x=[35], y=[15], z=[-35], mode='markers', marker=dict(color='green', size=5)))
                    fig.add_trace(go.Scatter3d(x=[35], y=[15], z=[-35], mode='markers', marker=dict(color='green', size=5))) #35n

                    arrow_length = 15
                    arrow_color = 'orange'
                    arrow_width = 3

                    for coord in red_coords:
                        start = np.array(coord)
                        camera_direction = np.array([0, 0, arrow_length]) @ rotation_matrix
                        end = start + camera_direction
                        fig.add_trace(create_3d_line(start, end, arrow_color, arrow_width))

                    fig.update_layout(scene=dict(xaxis_title='X Label', yaxis_title='Y Label', zaxis_title='Z Label'))

                    fig.show()
        

        #-------------------------------------------------------------------------------------------------------------------------------------------------------#

        
    # Display the frame with bounding boxes and class labels
    
    frame = cv2.resize(frame, (1000,900), interpolation=cv2.INTER_LINEAR)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()

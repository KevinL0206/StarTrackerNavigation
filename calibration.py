import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt


#--Camera Calibration--#    
            
# Chessboard dimensions (number of internal corners)
chessboard_rows = 7
chessboard_cols = 10

# Prepare object points (3D points in the real world space)
objp = np.zeros((chessboard_rows * chessboard_cols, 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_cols, 0:chessboard_rows].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images
objpoints = []  # 3D points in the real world space
imgpoints = []  # 2D points in the image plane

# Load and process the chessboard images
image_files = glob.glob("./*.jpg")  # Update the file path accordingly

scale_factor = 0.5
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

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
        img = cv2.drawChessboardCorners(img, (chessboard_cols, chessboard_rows), refined_corners * (1 / scale_factor), ret, (0, 255, 0), 10)

        
        h, w = img.shape[:2]
        

    

        # Display the original and undistorted images side by side
    
        plt.figure()
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.show()

        """
        scaled_img = cv2.resize(img, (1000,1000), interpolation=cv2.INTER_AREA)
        cv2.imshow('Corners', img)
        cv2.waitKey(0)
        """
        

        
        
        

#cv2.destroyAllWindows()

# Calibrate the camera
ret, camera_matrix, dist, rvecs1, tvecs1 = cv2.calibrateCamera(objpoints, imgpoints, resized_gray.shape[::-1], None, None)



# Display a visualization of the calibration
for fname in image_files:
    img = cv2.imread(fname)
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist, (w, h), 1, (w, h))

    # Undistort the image
    dst = cv2.undistort(img, camera_matrix, dist, None, newcameramtx)

    # Display the original and undistorted images side by side
   
    plt.figure()
    plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
    plt.show()


print(camera_matrix)
print("Camera Calibration Completed")
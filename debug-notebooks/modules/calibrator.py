import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from math import sqrt
from sklearn.model_selection import train_test_split

class Calibrator: 
    def __init__(self, flags_calib, image_path, board_points, image_size, pattern_size):
        self.flags_calib = flags_calib
        self.image_path = image_path

        self.board_points = board_points
        self.image_points = []
        self.world_points = []

        self.image_size = image_size
        self.pattern_size = pattern_size

        self.results = {
        "error_rms": None,
        "camera_matrix": None,
        "distortion_coeffs": None,
        "rvecs": None,
        "tvecs": None,
        "std_intrinsic": None,
        "std_extrinsic": None,
        "per_view_error": None
        }
        return
    
    def prepare_world_points(self):
        
        for i in range(len(self.image_points)):
            self.world_points.append(self.board_points)
        return
    
    def detect_using_SB(self):
        # Create empty variables
        images_detected = 0
        self.image_points = []
        self.world_points = []
        
        # For each image in directory
        for image_file in self.image_path:
            # Read the image file
            img = cv2.imread(image_file)
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Return the image size
            self.image_size = gray.shape[::-1]
            # Detect the corners in images
            detected, corners = cv2.findChessboardCornersSB(gray, self.pattern_size, None)
            
            if detected:
                # Save corners in image points array
                self.image_points.append(corners)
                self.world_points.append(self.board_points)
                images_detected += 1
                
        return images_detected
    
    def calibrate(self):
        # Calibrate the camera
        rms, camera_matrix, distortion_coeffs, rvecs, tvecs, std_intrinsic, std_extrinsic, per_view_error = cv2.calibrateCameraExtended(self.world_points, self.image_points, self.image_size, None, None, flags=self.flags_calib)  

        
        # Save the calibration results
        self.results = {
        "error_rms": rms,
        "camera_matrix": camera_matrix,
        "distortion_coeffs": distortion_coeffs,
        "rvecs": rvecs,
        "tvecs": tvecs,
        "std_intrinsic": std_intrinsic,
        "std_extrinsic": std_extrinsic,
        "per_view_error": per_view_error
        }

        #print("Result saved in 'Calibrator.results' ")
        
        return
    
    def calculate_error(self, type):
        # Create empty arrays for rotation and translation vectors
        rvecs = []
        tvecs = []
        errors = []
        
        # For each image in directory
        for image_file in self.image_path:
            # Read the image file
            img = cv2.imread(image_file)
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Return the image size
            self.image_size = gray.shape[::-1]
            # Detect the corners in images
            detected, corners = cv2.findChessboardCornersSB(gray, self.pattern_size, None)

            # If it was detected
            if detected == True:
                # Calculate extrinsic parameters 
                _, rvec, tvec = cv2.solvePnP(self.board_points, corners, self.results['camera_matrix'], self.results['distortion_coeffs'])
                rvecs.append(rvec)
                tvecs.append(tvecs)

                # Calculate projected image points
                projected_image_points, _ = cv2.projectPoints(self.board_points, rvec, tvec, self.results['camera_matrix'], self.results['distortion_coeffs'])
                
                # Find the Euclidean Distance between projected and detected image points
                if type == 'mean':
                    error = cv2.norm(corners, projected_image_points, normType= cv2.NORM_L2) / len(projected_image_points)
                
                elif type == 'rms':
                    error = cv2.norm(corners, projected_image_points, normType= cv2.NORM_L2) / sqrt(len(projected_image_points))

                errors.append(error)
        return errors
    
    def per_view_error(self):
        image_files = []
        
        # Extract the image file name
        for image in self.image_path:
            nome_arquivo = os.path.basename(image)
            image_files.append(nome_arquivo)

        # Plot the images errors
        plt.figure(figsize=(10,7))
        plt.bar(image_files, self.results['per_view_error'].flatten(), color = 'royalblue', width=.75)

        plt.axhline(y=self.results['error_rms'], color='gray', linestyle='--', linewidth=1)
        
        plt.xticks(rotation=45, ha='right')  # Rotacionar os rótulos das imagens para melhor visualização
        plt.tight_layout()  # Ajusta o layout para se encaixar bem na figura
        plt.show()
        
            
    def train_test_split_images(self):
        
        # Step 1 - Initial Calibration
        self.calibrate()
        erros = self.calculate_error('rms')
        median = np.median(erros)
        std = np.std(erros)
        
        # Step 2 - Remove Outliers
        outliers = []
        limiar = median + std
        error = self.results['per_view_error'].flatten()
        
        # Check if the image error is gratter than limiar
        for idx, image in enumerate(self.image_path):
            
            if error[idx] > limiar:
                outliers.append(image)
                
        # Remove outliers from image path    
        for o in outliers:
            print('Remove outlier - ',os.path.basename(o))
            self.image_path.remove(o)

        # Train - Test split
        images_train, images_test, _, _ = train_test_split(self.image_path, np.zeros(len(self.image_path)), test_size = 0.3)
        
        return images_train, images_test
'''
Script to generate random images for camera 
calibration with a 7x7 calibration pattern

'''

import time
import math
import numpy as np
import cv2
import random
from scipy.spatial.transform import Rotation
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

# Connect and configure the simulation 
client = RemoteAPIClient()
sim = client.getObject('sim')
defaultIdleFps = sim.getInt32Param(sim.intparam_idle_fps)   
sim.setInt32Param(sim.intparam_idle_fps, 0)
patternSize = (7,7)

# Get the vision sensor handle
visionSensorHandle = sim.getObject('/Vision_sensor')
plano = sim.getObject('/plano')
#%%
def randt(L): # Confined in a cube with an edge of length L
    return [2*L*random.random()-L for _ in range(3)]
    
def sum_coord(A, B):
    for coord in range(len(A)):
        B[coord] += A[coord]
    return B

# Randomly translate the calibration pattern
def rand_translation(L, position, handle = sim.getObject('/plano')):

    ds = randt(L)
    sim.setObjectPosition(handle, -1, sum_coord(position, ds))

# Randomly rotate the calibration pattern
def randR():
    # Initialize empty vector
    rotation = np.array([0,0,0])

    # For each rotation axis
    for axis in ['x', 'y', 'z']:

        # Find a random euler angle
        r = Rotation.from_euler(axis,  np.random.randint(0, 20), degrees=True).as_euler('xyz', degrees=True)
        rotation = rotation + r
    
    # Return in the Coppelia Format
    return (rotation * np.pi / 180).tolist()
#%% Main
print('Program started')


# Start simulation in CoppeliaSim
sim.startSimulation()

# Number of calibration images
number_images = 10

# Capture Loop
while (t := sim.getSimulationTime()) < 10:
    p=sim.getObjectPosition(plano,-1)

    for idx in range(number_images):
        # New aleatory configuration
        sim.setObjectOrientation(plano, plano, randR())
        rand_translation(0.3, p)

        # Take a photo
        img, resX, resY = sim.getVisionSensorCharImage(visionSensorHandle)
        img = np.frombuffer(img, dtype=np.uint8).reshape(resY, resX, 3)
        img = cv2.flip(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 0)

        # Check if are visible corners
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detected, _ = cv2.findChessboardCornersSB(gray, patternSize, None)
        
        if detected == True:
            print(f"Corners detected in image {idx}")
                
            # Write the image
            cv2.imwrite(f'image4{idx}.jpg',img)
            time.sleep(0.75)

sim.stopSimulation()

# Restore the original idle loop frequency:
sim.setInt32Param(sim.intparam_idle_fps, defaultIdleFps)

cv2.destroyAllWindows()

print('Program ended')
## This script generates artificial 2D MOT data for testing purposes.

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage.filters as filters
import seaborn as sns
import pandas as pd

import math
import emcee

from params_MOT.io_package import get_data_file_path, load_data, load_image
import params_MOT as pm

# Define image size, starting vectors
image_size=50
x=np.linspace(1,image_size,image_size)
y=np.linspace(1,image_size,image_size)
x,y=np.meshgrid(x, y)

# 2D Gaussian image model
theta = (image_size/2,image_size/2,400,image_size/7.5,image_size/9,0,0,0)
image = pm.MOT_bare_model(x,y,theta)

# 2D Gaussian image model + CCD readout charge noise
image = pm.Image_with_CCD_readout_charge(image, 40)

# Add two other sources of noise: the background scattered light which is detected by the PMTs, which also have some Poissonian noise in them:
image_poisson = image + pm.detected(pm.background(image_size,0,10000))-pm.detected(pm.background(image_size,0,10000))

# Show the image:
plt.figure(1)
plt.imshow(image_poisson,cmap="jet", interpolation='none')
plt.colorbar()

# Save generated image to csv file:
time = 100 # (ms) random choice
np.savetxt("model_data.csv", image, delimiter=" ")
## This script is similar to model.ipynb, only that it is a runnable as a Python script. In this version, for now we are only using MCMC to fit to generated data. 

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

## Performing Bayesian inference using MCMC for marginalization

# We load the sample data that we just generated:
model_data_path = get_data_file_path('model_data.csv')
model_data = load_data(data_file = model_data_path, delim = ' ')

image_size = 50 

image_object = load_image(model_data)

# Run the MCMC:
initial_guess = [25, 25, 400, 6.6667, 5.5556, 100, 20, 20] # from HBL figure 1 and randomly guessing
emcce_sample = pm.sampler(model_data, 8, 50, 2000,image_size,initial_guess)

# Show the results:
fig, (ax_sigma_x, ax_sigma_y, ax_sigma_m, ax_sigma_g) = plt.subplots(4)
ax_sigma_x.set(ylabel='sigma_x')
ax_sigma_y.set(ylabel='sigma_y')
ax_sigma_m.set(ylabel='sigma_m')
ax_sigma_g.set(ylabel='sigma_g')

for i in range(20):
    sns.tsplot(emcce_sample.chain[i,:,3], ax=ax_sigma_x)
    sns.tsplot(emcce_sample.chain[i,:,4], ax=ax_sigma_y)
    sns.tsplot(emcce_sample.chain[i,:,6], ax=ax_sigma_m)
    sns.tsplot(emcce_sample.chain[i,:,7], ax=ax_sigma_g)

# Throw away first 1000 steps
ndim = 8

samples = emcce_sample.chain[:,1000:,:]
traces = samples.reshape(-1, ndim).T

parameter_samples = pd.DataFrame({'sigma_x': traces[3], 'sigma_y': traces[4]})

q = parameter_samples.quantile([0.16,0.50,0.84], axis=0)
sigma_x = q['sigma_x'][0.50]
sigma_y = q['sigma_y'][0.50]

print(q)
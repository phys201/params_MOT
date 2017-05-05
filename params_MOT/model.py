import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage.filters as filters
import seaborn as sns
import pandas as pd

import math
import emcee

from params_MOT.io_package import get_data_file_path, load_data, load_time, load_power
import params_MOT as pm
from params_MOT import MOT_image

## Performing Bayesian inference using MCMC for marginalization
def find_params_MOT(data_file_name, data_dir='data', image_size = 50, mc_params=(200, 1000, 400),initial_guess=[25, 25, 400, 6.6667, 5.5556, 100, 20, 20], suppressMessages = False):
	'''
	Function to load data based on the file name specified in data_file_name, find parameters based on Bayesian inference using MCMC for parameter marginalization.

	Returns an array of with the MOT_image object corresponding to the data, and with a pandas object containing the inferred parameter sigma_x and sigma_y, at the .16, .5, an ,84 quantile

	Keyword arguments:
	data_file_name		-- String denoting filename of data
	data_dir			-- String denoting name of data directory
	image_size			-- Size of MOT image (which is assumed to be a square).
	mc_params			-- duplet of MCMC parameters: (number of walkers, number of steps, burn_in_steps).
						-- walkers = individual traces in the Monte Carlo algorithm
						-- steps = length of said traces
						-- burn_in_steps = steps after which the trace settles around a value
	initial_guess		-- tuple of initial MCMC guesses, consisting of (center_x, center_y, amplitude, sigma_x, sigma_y, background_offset, sigma_m, sigma_g).
	suppressMessages		-- Boolean which indicates whether or not messages, including plots, should be output.
	'''

	if(not suppressMessages):
		print('Loading data...')
	real_data_path = get_data_file_path(data_file_name, data_dir)
	data = load_data(data_file = real_data_path, delim = ',')

	image_object = MOT_image.MOT_image(data, load_time(data_file_name), load_power(data_file_name), image_size = image_size) # load image of MOT data
	if (not suppressMessages):
		image_object.show(gauss_filter = True) # display image

	if (not suppressMessages):
		print("The life time of and the laser power used on the MOT are %s ms and 1/%s (fractional units out of the max power, which is 60mW per beam)" %(image_object.time, image_object.power))


	(nwalkers, nsteps, burn_in_steps) = mc_params
	ndim = 8 # normally 8 parameters to be fitted in our model	

	if (not suppressMessages):
		print('Data loaded. Running emcee sampler...')
	emcee_sample = pm.sampler(data, ndim, nwalkers, nsteps,image_size,initial_guess)

	if (not suppressMessages):
		print('Emcee finished. Generating plots...')

	# Show the results:
	if (not suppressMessages):
		fig, (ax_center_x, ax_center_y, ax_amplitude, ax_sigma_x, ax_sigma_y, ax_background_offset, ax_sigma_m,ax_readout_charge) = plt.subplots(8)
		ax_center_x.set(ylabel='center_x')
		ax_center_y.set(ylabel='center_y')
		ax_amplitude.set(ylabel='amplitude')
		ax_sigma_x.set(ylabel='sigma_x')
		ax_sigma_y.set(ylabel='sigma_y')
		ax_background_offset.set(ylabel='background_offset')
		ax_sigma_m.set(ylabel='sigma_m')
		ax_readout_charge.set(ylabel='readout_charge')

		for i in range(20):
			sns.tsplot(emcee_sample.chain[i,:,0], ax=ax_center_x)
			sns.tsplot(emcee_sample.chain[i,:,1], ax=ax_center_y)
			sns.tsplot(emcee_sample.chain[i,:,2], ax=ax_amplitude)
			sns.tsplot(emcee_sample.chain[i,:,3], ax=ax_sigma_x)
			sns.tsplot(emcee_sample.chain[i,:,4], ax=ax_sigma_y)
			sns.tsplot(emcee_sample.chain[i,:,5], ax=ax_background_offset)
			sns.tsplot(emcee_sample.chain[i,:,6], ax=ax_sigma_m)
			sns.tsplot(emcee_sample.chain[i,:,7], ax=ax_readout_charge)
		
	# Throw away first 1000 steps and determine parameters based on 50th percentile in the fit
	ndim = 8

	samples = emcee_sample.chain[:,burn_in_steps:,:]
	traces = samples.reshape(-1, ndim).T

	parameter_samples = pd.DataFrame({'sigma_x': traces[3], 'sigma_y': traces[4]})

	q = parameter_samples.quantile([0.16,0.50,0.84], axis=0)
	sigma_x = q['sigma_x'][0.50]
	sigma_y = q['sigma_y'][0.50]

	if (not suppressMessages):
		print(q)
	
		joint_kde = sns.jointplot(x='sigma_x', y='sigma_y', data=parameter_samples, kind='kde')

	return [image_object, q]

def gen_model_data(data_file_name, image_size, theta, ccdnoise, background_lv):
	'''
	Function to generate artificial data for our MOT model. Incorporates the following sources of uncertainties: Gaussian uncertainty in the model, CCD readout shot noise, background scattered light.
	
	Keyword arguments:
	data_file_name		-- Name of data file for which artificial data will be saved.
	theta				-- model parameter array (center_x, center_y, amplitude, sigma_x, sigma_y, background_offset, sigma_m, sigma_g)
	ccdnoise			-- parameter controlling ccdnoise level
	background_lv		-- parameter controlling scattered background light noise level
	image_size			-- size of MOT image
	'''
	# Define starting vectors
	x=np.linspace(1,image_size,image_size)
	y=np.linspace(1,image_size,image_size)
	x,y=np.meshgrid(x, y)

	# 2D Gaussian image model
	image = pm.MOT_bare_model(x,y,theta)

	# 2D Gaussian image model + CCD readout charge noise
	image = pm.Image_with_CCD_readout_charge(image, 40)

	# Add two other sources of noise: the background scattered light which is detected by the PMTs, which also have some Poissonian noise in them:
	image_poisson = image + pm.detected(pm.background(image_size,0,background_lv))-pm.detected(pm.background(image_size,0,background_lv))

	# Show the image:
	plt.figure(1)
	plt.imshow(image_poisson,cmap="jet", interpolation='none')
	plt.colorbar()

	# Save generated image to csv file:
	np.savetxt(data_file_name, image, delimiter=",")
	return image

def find_params_MOTs(list_data_files, data_dir='data', image_size = 50, mc_params=(200, 1000, 400),initial_guess=[25, 25, 400, 6.6667, 5.5556, 100, 20, 20], suppressMessages = True):
	'''
		Function to return MOT_image objects and their inferred sigma_x and sigma_y for a list of files.

		Keyword arguments:
		list_data_files		-- Array of data file names for which we find the parameters.
		data_dir			-- String denoting name of data directory
		image_size			-- Size of MOT image (which is assumed to be a square).
		mc_params			-- duplet of MCMC parameters: (number of walkers, number of steps, burn_in_steps).
							-- walkers = individual traces in the Monte Carlo algorithm
							-- steps = length of said traces
							-- burn_in_steps = steps after which the trace settles around a value
		initial_guess		-- tuple of initial MCMC guesses, consisting of (center_x, center_y, amplitude, sigma_x, sigma_y, background_offset, sigma_m, sigma_g).
		suppressMessages		-- Boolean which indicates whether or not messages, including plots, should be output.
	'''

	q = []
	for f in list_data_files:
		q.append(find_params_MOT(f, data_dir, image_size, mc_params, initial_guess, suppressMessages))

	return q
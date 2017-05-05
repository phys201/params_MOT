import unittest
import numpy as np
import params_MOT as pm
from params_MOT.io_package import *
from params_MOT.model import *
import os

# Variables to be used in tests:
image_size = 10
theta = [image_size/2,image_size/2,400,image_size/7.5,image_size/9,1,1,1]
x = np.linspace(1,image_size,image_size)
y = np.linspace(1,image_size,image_size)
data = np.random.rand(image_size,image_size)
initial_guess = [25, 25, 400, 6.6667, 5.5556, 100, 20, 20]
filename=('test_1_8power.csv')

class TestParams_MOT(unittest.TestCase):
	'''
	Class containing unit tests for params_MOT package. This tests the major functions of the package, from basic models of noise and the MOT, functions for Bayesian inference, input/output functions, and the MCMC sampler functions. They verify whether the functions are still running.
	'''
	'''
	Tests for basic model functions
	'''
	
	def test_gaussian1d(self):
		self.assertTrue=(pm.gaussian_1d(1,1,1,1))
	def test_gaussian2d(self):
		self.assertTrue=(pm.gaussian_2d(1,-2,0,0,1,1,1))
	def test_image_with_ccd_readout_charge(self):
		self.assertTrue=(pm.Image_with_CCD_readout_charge(data,40))
	def test_MOT_bare_model(self):
		self.assertTrue=(pm.MOT_bare_model(np.random.rand(1,image_size),np.random.rand(1,image_size),theta))

	'''
	Tests for Bayesian inference: prior, likelihood, posterior
	'''
	def test_prior(self):
		self.assertTrue=(pm.log_prior(theta))
	def test_likelihood(self):
		self.assertTrue=(pm.log_likelihood(theta,x,y,data))
	def test_posterior(self):
		self.assertTrue=(pm.log_posterior(theta, x,y,data))
	
	'''
	Tests for input and output of data
	'''
	def test_load_data(self):
		'''test_load_data: This uses the function gen_model_data to generate a test data file which is then loaded and tested. The filename is also parsed for time and power.'''
		gen_model_data(filename,image_size,theta,40,10000) # Create model data file
		mot_data = load_data(filename, delim=',')
		self.assertTrue(np.sum(mot_data))
		self.assertTrue(load_power(filename))
		self.assertTrue(load_time(filename))
		os.remove(filename) # Remove file after test
	
	'''
	Tests for model inference
	'''
	def test_sampler(self):
		'''test_sampler: This tests the emcee sampler on artificial data generated using gen_model_data. The image size is made small and the number of steps also small to make the tests run faster.'''
		image = gen_model_data(filename,image_size,theta,40,10000)
		self.assertTrue(pm.sampler(image, 8, 50, 200,image_size,initial_guess))
		os.remove(filename)
	def test_find_params_MOT(self):
		'''test_find_params_MOT: This is a more extensive test than test_sampler. It generates artificial model data, sets up and runs the MCMC sampler, and returns back the parameters of the model.'''
		gen_model_data(filename,image_size,theta,40,10000)
		self.assertTrue(find_params_MOT(filename,'.',image_size,(50,200,50),initial_guess,suppressMessages=False))
		
if __name__ == '__main__':
	unittest.main()
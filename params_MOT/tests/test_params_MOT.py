import unittest
import numpy as np
from params_MOT import params_MOT as pm
from io_package import *
import os

class BasicFunctionsTestCase(unittest.TestCase):
	"Tests for basic functions in __init__.py of params_MOT. Ensures that each function returns a number/array instead of an error."
	def test_gaussian1d(self):
		self.assertTrue=(pm.gaussian_1d(1,1,1,1))
	def test_gaussian2d(self):
		self.assertTrue=(pm.gaussian_2d(1,-2,0,0,1,1,1))
	def test_image_with_ccd_readout_charge(self):
		self.assertTrue=(pm.Image_with_CCD_readout_charge(np.random.rand(50,50),40))
	def test_MOT_bare_model(self):
		self.assertTrue=(pm.MOT_bare_model(np.random.rand(1,50),np.random.rand(1,50),[40/2,40/2,400,40/7.5,40/9,0,0,0]))
	def test_MOT_model(self):
		self.assertTrue=(pm.MOT_model(np.random.rand(1,50),np.random.rand(1,50),[40/2,40/2,400,40/7.5,40/9,0,0,0]))
	def test_likelihood(self):
		self.assertTrue=(pm.log_likelihood([40/2,40/2,400,40/7.5,40/9,1,1,1],np.linspace(1,50,50),np.linspace(1,50,50),np.random.rand(50,50)))
	def test_prior(self):
		self.assertTrue=(pm.log_prior([40/2,40/2,400,40/7.5,40/9,1,1,1]))
	def test_posterior(self):
		self.assertTrue=(pm.log_posterior([40/2,40/2,400,40/7.5,40/9,1,1,1], np.linspace(1,50,50), np.linspace(1,50,50), np.random.rand(50,50)))
	def test_load_data(self):
		mot_data = load_data('model_data.csv', delim=' ')
		self.assertTrue(np.sum(mot_data))
	def test_load_image(self):
		self.assertTrue(load_image(np.random.rand(50,50),50))
	def test_sampler(self):
		self.assertTrue(pm.sampler(np.random.rand(50,50), 8, 50, 2000,50))
if __name__ == '__main__':
	unittest.main()
import unittest
import numpy as np

import params_MOT as pm

class BasicFunctionsTestCase(unittest.TestCase):
	"Tests for basic functions in __init__.py of params_MOT. Ensures that each function returns a number/array instead of an error."
	def test_gaussian1d(self):
		self.assertTrue=(pm.gaussian_1d(1,1,1,1))
	def test_gaussian2d(self):
		self.assertTrue=(pm.gaussian_2d(1,-2,0,0,1,1,1))
	def test_background(self):
		self.assertTrue=(pm.background(1,2,3))
	def model(self):
		self.assertTrue=(pm.model(1,1,1))
	def test_motmodel(self):
		self.assertTrue=(pm.MOTmodel(1,1,1,1,1,1,1))
	def test_detected(self):
		self.assertTrue=(pm.detected(1))
if __name__ == '__main__':
	unittest.main()
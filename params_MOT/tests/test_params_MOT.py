from unittest import TestCase
import numpy as np

import params_MOT as pm

class BasicFunctionsTestCase(TestCase):
	"Tests for basic functions in __init__.py of params_MOT"
	def test_is_gaussian(self):
		"Test if gaussian returns a number"
		self.assertTrue=(pm.gaussian_2d(1,-2,0,0,1,1,1))
		
if __name__ == '__main__':
	unittest.main()
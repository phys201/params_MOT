# This contains functions from other parts of the package which were deemed non-essential for now but we keep in case we need them again during the development process. When the package has been tested and shown to not need these functions then we will permanently delete them.

def MOT_model(x, y, theta):
    # Use 40 for the readout_charge for now
    return Image_with_CCD_readout_charge(MOT_bare_model(x, y, theta), 40)

# From tests (as MOT_model is now deleted)
def test_MOT_model(self):
	self.assertTrue=(pm.MOT_model(np.random.rand(1,50),np.random.rand(1,50),[40/2,40/2,400,40/7.5,40/9,0,0,0]))
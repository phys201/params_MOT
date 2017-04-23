# This contains functions from other parts of the package which were deemed non-essential for now but we keep in case we need them again during the development process. When the package has been tested and shown to not need these functions then we will permanently delete them.

def MOT_model(x, y, theta):
    # Use 40 for the readout_charge for now
    return Image_with_CCD_readout_charge(MOT_bare_model(x, y, theta), 40)

def background(image_size,offset,scattered_light):
    N=image_size
    return np.add([[scattered_light for i in range(N)] for j in range(N)],offset)

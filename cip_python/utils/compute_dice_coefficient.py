import numpy as np
from scipy.stats import mode, kurtosis, skew
from optparse import OptionParser
import nrrd
import pdb

def compute_dice_coefficient(lm1, lm2, radius):
    """

    Parameters
    ----------
    lm1 : array, shape ( X, Y, Z )
        The first label map.
    
    lm2 : array, shape ( X, Y, Z )    
        The second label map.

    radius : int, optional
        Integer value greater than or equal to 0. If specified, for a given 
        voxel location, a search will be performed in the other label map
        within a neighborhood with this radius. 0 by default.
    
    Returns
    -------
    dice : float
        The Dice coefficient, computed as 2*(A&B)/(A+B), where A&B is the number
        of overlapping voxels, A is the number of voxels in lm1 and B is the
        number of voxels in lm2. The Dice coefficient is between 0 and 1, with 1
        indicating perfect agreement, and 0 indicating no agreement.
    """
    assert lm1.shape[0] == lm2.shape[0] and lm1.shape[1] == lm2.shape[1] and \
      lm1.shape[2] == lm2.shape[2], "Label maps are not the same size"
      
    X = lm1.shape[0]
    Y = lm1.shape[1]
    Z = lm1.shape[2]
      
    lm1_x, lm1_y, lm1_z = np.where(lm1 > 0)
    lm2_x, lm2_y, lm2_z = np.where(lm2 > 0)    

    if lm1_x.shape[0] == 0 and lm2_x.shape[0] == 0:
        return 1.0

    intersection = 0    
    for n in xrange(0, lm1_x.shape[0]):
        val = lm1[lm1_x[n], lm1_y[n], lm1_z[n]]
        if val in \
               lm2[np.max([0, lm1_x[n]-radius]):np.min([lm1_x[n]+radius+1, X]),
                   np.max([0, lm1_y[n]-radius]):np.min([lm1_y[n]+radius+1, Y]),
                   np.max([0, lm1_z[n]-radius]):np.min([lm1_z[n]+radius+1, Z])]:
            intersection += 1

    dice = 2.*intersection/(lm1_x.shape[0] + lm2_x.shape[0])
    return dice
    
if __name__ == "__main__":
    desc = """Compute the Dice coefficient between two label maps"""
    
    parser = OptionParser(description=desc)
    parser.add_option('--lm1', help='The first label map', 
                      dest='lm1', metavar='<string>', default=None)
    parser.add_option('--lm2', help='The second label map',
                      dest='lm2', metavar='<string>', default=None)    
    parser.add_option('--radius', help='Integer value greater than or equal \
    to 0. If specified, for a given voxel location, a search will be performed \
    in the other label map within a neighborhood with this radius. 0 by \
    default.', dest='radius', metavar='<integer>', default=0)    
    
    (options, args) = parser.parse_args()

    lm1, lm1_header = nrrd.read(options.lm1)
    lm2, lm2_header = nrrd.read(options.lm2)    

    dice = compute_dice_coefficient(lm1, lm2, int(options.radius))
    print ("Dice coefficient: %s" % dice)
    

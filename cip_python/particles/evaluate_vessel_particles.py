import os
import pdb
import numpy as np
from optparse import OptionParser
import vtk
import nrrd
from skimage import morphology
from cip_python.particles.chest_particles import ChestParticles

def evaluate_vessel_particles(skel, poly, spacing, origin, irad):
    """Computes a measure of how well a particles data set samples and 
    underlying structure (vessel tree) represented by a skeletonized
    version of the vessel mask.

    Parameters
    ----------
    skel : array , shape : (L, M, N)
        Skeletonized mask of underlying vessel tree. Particles should be in 
        close proximity to this mask

    poly : vtk polydata
        Particles polydata file

    spacing : list
        Physical spacing between the voxels in skel in the x, y, and z 
        directions
        
    origin : list
        3D coordinate skel's origin     
    
    irad : float
        The irad parameter used when generating the vessel particles

    Returns
    -------
    score : float
        The final score in the interval [0, 1]. The higher the better.
    
    """
    delt_x = int(np.ceil(irad/spacing[0]))
    delt_y = int(np.ceil(irad/spacing[1]))
    delt_z = int(np.ceil(irad/spacing[2]))      

    # Loop over all particles. For each particle, we want to create a map
    # between the image index that corresponds to the particle's location and
    # the particle ID. This map will be used later to tally up the FNs. Note
    # that we assume the same coordinate system for the particles and the
    # skeleton mask. Also in this loop, we will tally up the TPs and FPs
    # by looking around the skeleton mask within an neighborhood determined
    # by irad (neighborhood size: delt_x, delt_y, delt_z). If we see a skeleton
    # voxel in this neighborhood, then we have a TP. Otherwise, we have a 
    # FP
    TP = 0
    FP = 0
    FN = 0
    
    index_id_map = {}
    for p in xrange(0, poly.GetNumberOfPoints()):
        i = int(np.round((poly.GetPoint(p)[0] - origin[0])/spacing[0]))
        j = int(np.round((poly.GetPoint(p)[1] - origin[1])/spacing[1]))
        k = int(np.round((poly.GetPoint(p)[2] - origin[2])/spacing[2]))              
        index_id_map[str(i)+','+str(j)+','+str(k)] = p

        found = False
        for x in xrange(i-delt_x, i+delt_x+1):            
            for y in xrange(j-delt_y, j+delt_y+1):
                for z in xrange(k-delt_z, k+delt_z+1):
                    if x > 0 and x < skel.shape[0] and \
                      y > 0 and y < skel.shape[1] and \
                      z > 0 and z < skel.shape[2]:
                      if skel[x, y, z] > 0:
                          found = True
        if found:
            TP += 1
        else:
            FP += 1


    # Now we need to compute the FNs by finding all the skeleton indices and
    # looking around the particles dataset to find any particles in the
    # neighborhood. If there are none, then we have a FN.
    ids = np.where(skel[:,:,:]>0)
    for n in xrange(0, ids[0].shape[0]):
        i = ids[0][n]
        j = ids[1][n]
        k = ids[2][n]

        found = False
        for x in xrange(i-delt_x, i+delt_x+1):            
            for y in xrange(j-delt_y, j+delt_y+1):
                for z in xrange(k-delt_z, k+delt_z+1):
                    key = str(x) + ',' + str(y) + ',' + str(z)
                    if key in index_id_map.keys():
                        found = True

        if not found:
            FN += 1

    score = 2.*TP/(FP + FN + 2*TP)
    return score
    
if __name__ == "__main__":  
    desc = "Evaluate how well a particles data set samples an underlying \
    structure using the skeletonized mask of that structure. The measure \
    is computed as 2TP/(FP + FN + 2*TP), where TP is the true positive count, \
    FP is the false positive count, and FN is the false negative count."
    
    parser = OptionParser(desc)
    parser.add_option("-p", help='Particles file name', dest="particles")
    parser.add_option("-s", help='Mask of the skeleton corresponding to \
                      vessel tree', dest="mask")
    parser.add_option("--irad", help='The inter-particle distance that \
                      was specified when generating the particles', 
                      dest="irad")
  
    (op, args) = parser.parse_args()

    irad = float(op.irad)
    
    skel, skel_header = nrrd.read(op.mask)

    origin = skel_header['space origin']
    spacing = []
    spacing.append(skel_header['space directions'][0][0])
    spacing.append(skel_header['space directions'][1][1])
    spacing.append(skel_header['space directions'][2][2])        
    
    particles_reader = vtk.vtkPolyDataReader()
    particles_reader.SetFileName(op.particles)
    particles_reader.Update()

    poly = particles_reader.GetOutput()

    score = evaluate_vessel_particles(skel, poly, spacing, origin, irad)
    print "Score: %s" % score
    



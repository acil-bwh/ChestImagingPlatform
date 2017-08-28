from argparse import ArgumentParser
import vtk, pdb
import numpy as np

def unit_vector(vector):
    """ Returns the unit vector of the vector."""
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in degrees between vectors 'v1' and 'v2' """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    
    return (180./np.pi)*np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def transfer_particles_region_type_values(unlabeled_particles, labeled_particles,
    num_candidates=5, dist_thresh=1., v_angle_thresh=None, a_angle_thresh=None,
    f_angle_thresh=None, scale_thresh=None):
    """

    Parameters
    ----------
    unlabeled_particles : vtkPolyData
        Unlabeled particles

    labeled_particles : vtkPolyData
        Labeled particles

    num_candidates : int, optional
        For a given particle in the unlabeled data set, this is the number of
        candidate particles in the labeled data set to consider as possible
        matches. The best match in terms of distance and orientation will be
        used to transfer the label.

    dist_thresh : float, optional
	    A particle in the labeled data set must be at least this close (in mm)
        to a particle in the unlabeled data set to be considered for label
        transfer

    v_angle_thresh : float, optional
        Vessel angle threshold (in degrees). The orientation of a candidated
        labeled particle must deviate by no more than this amount to be
        considered for label transfer

    a_angle_thresh : float, optional
        Airway angle threshold (in degrees). The orientation of a candidated
        labeled particle must deviate by no more than this amount to be
        considered for label transfer

    f_angle_thresh : float, optional
        Fissure angle threshold (in degrees). The orientation of a candidated
        labeled particle must deviate by no more than this amount to be
        considered for label transfer                

    scale_thresh : float, optional                
	    Scale difference threshold. The difference in scale between a
        candidated labeled particle and an unlabeled particle must be no
        greater than this amount to be considered for label transfer

    Returns
    -------
    out_particles : vtkPolyData
        Equivalent to unlabeled_particles, except the output particles data set
        will have a point data array called ChestRegionChestType with the
        values set based on the value transfer rules in this function.    
    """
    locator = vtk.vtkPointLocator()
    locator.SetDataSet(labeled_particles)
    locator.BuildLocator()

    ids = vtk.vtkIdList()
    for i in xrange(0, unlabeled_particles.GetNumberOfPoints()):
        pt_unlabeled = np.array(unlabeled_particles.GetPoint(i))
        scale_unlabeled = unlabeled_particles.GetPointData().\
            GetArray('scale').GetTuple(i)[0]
        hevec0_unlabeled = np.array(unlabeled_particles.GetPointData().\
            GetArray('hevec0').GetTuple(i))
        hevec1_unlabeled = np.array(unlabeled_particles.GetPointData().\
            GetArray('hevec1').GetTuple(i))
        hevec2_unlabeled = np.array(unlabeled_particles.GetPointData().\
            GetArray('hevec2').GetTuple(i))

        transfer_label = None
        min_dist =  np.finfo('d').max    
        locator.FindClosestNPoints(num_candidates, pt_unlabeled, ids)
        for n in xrange(0, num_candidates):
            pt_labeled = \
              np.array(labeled_particles.GetPoint(ids.GetId(n)))
            scale_labeled = labeled_particles.GetPointData().\
              GetArray('scale').GetTuple(ids.GetId(n))[0]
            hevec0_labeled = np.array(labeled_particles.GetPointData().\
              GetArray('hevec0').GetTuple(ids.GetId(n)))
            hevec1_labeled = np.array(labeled_particles.GetPointData().\
              GetArray('hevec1').GetTuple(ids.GetId(n)))
            hevec2_labeled = np.array(labeled_particles.GetPointData().\
              GetArray('hevec2').GetTuple(ids.GetId(n)))
            dist = np.sqrt(np.sum((pt_unlabeled - pt_labeled)**2))
    
            if dist <= dist_thresh:
                candidate = True
                if scale_thresh is not None and np.abs(scale_unlabeled - \
                    scale_labeled) > scale_thresh:
                    candidate = False
                if v_angle_thresh is not None:
                    if angle_between(hevec0_labeled, hevec0_unlabeled) > \
                      v_angle_thresh:
                        candidate = False
                if f_angle_thresh is not None:
                    if angle_between(hevec2_labeled, hevec2_unlabeled) > \
                      f_angle_thresh:
                        candidate = False
                if a_angle_thresh is not None:
                    if angle_between(hevec1_labeled, hevec1_unlabeled) > \
                      a_angle_thresh:
                        candidate = False
                if scale_thresh is not None:
                    if np.abs(scale_unlabeled - scale_labeled) > scale_thresh:
                        candidate = False                

                if candidate and dist < min_dist:
                    min_dist = dist
                    transfer_label = labeled_particles.GetPointData().\
                        GetArray('ChestRegionChestType').GetTuple(ids.GetId(n))
                        
        if transfer_label is not None:
            unlabeled_particles.GetPointData().\
              GetArray('ChestRegionChestType').SetTuple(i, transfer_label)        

    return unlabeled_particles
                  
if __name__ == "__main__":    
    desc = """Transfers ChestRegionChestType array labels from one particle
    data set (labeled) to another (unlabeled)."""
        
    parser = ArgumentParser(description=desc)
    parser.add_argument("--up", help='Unlabeled particles',
                        dest="unlabeled_particles")
    parser.add_argument("--lp", help='Labeled particles',
                        dest="labeled_particles")
    parser.add_argument("--op", help='Output particles', dest="out_particles")
    parser.add_argument("--n", help='For a given particle in the unlabeled \
    data set, this is the number of candidate particles in the labeled data \
    set to consider as possible matches. The best match in terms of distance \
    and orientation will be used to transfer the label', dest="num_candidates",
    default=5)
    parser.add_argument("--dist", help='A particle in the labeled data set \
    must be at least this close (in mm) to a particle in the unlabeled data \
    set to be considered for label transfer', dest="dist_thresh", default=1.)
    parser.add_argument("--vat", help='Vessel angle threshold (in degrees). \
    The orientation of a candidated labeled particle must deviate by no more \
    than this amount to be considered for label transfer.',
    dest="v_angle_thresh", default=None)
    parser.add_argument("--fat", help='Fissure angle threshold (in degrees). \
    The orientation of a candidated labeled particle must deviate by no more \
    than this amount to be considered for label transfer',
    dest="f_angle_thresh", default=None)
    parser.add_argument("--aat", help='Airway angle threshold (in degrees). \
    The orientation of a candidated labeled particle must deviate by no more \
    than this amount to be considered for label transfer',
    dest="a_angle_thresh", default=None)
    parser.add_argument("--st", help='Scale difference threshold. The \
    difference in scale between a candidated labeled particle and an \
    unlabeled particle must be no greater than this amount to be considered \
    for label transfer', dest="scale_thresh", default=None)
    
    op = parser.parse_args()

    print "Reading unlabeled particles..."    
    unlabeled_reader = vtk.vtkPolyDataReader()
    unlabeled_reader.SetFileName(op.unlabeled_particles)
    unlabeled_reader.Update()

    print "Reading labeled particles..."    
    labeled_reader = vtk.vtkPolyDataReader()
    labeled_reader.SetFileName(op.labeled_particles)
    labeled_reader.Update()

    print "Transferring chest-region chest-type values..."
    out_particles = transfer_particles_region_type_values(\
        unlabeled_reader.GetOutput(), labeled_reader.GetOutput(),
        op.num_candidates, op.dist_thresh, op.v_angle_thresh,
        op.a_angle_thresh, op.f_angle_thresh, op.scale_thresh)

    print "Writing particles..."
    writer = vtk.vtkPolyDataWriter()
    writer.SetInput(out_particles)
    writer.SetFileName(op.out_particles)
    writer.Update()

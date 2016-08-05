import numpy as np
import vtk
from numpy import linalg as LA


class ParticleMetrics:
    """Handles computation of metrics used to compare two particles data sets.

    Parameters
    ----------
	test_particles : vtkPolyData
	    The particles data set to be evaluated.

	ref_particles : vtkPolyData
	    The reference particles data set to compare against.

    particle_type : string, optional
        Either 'vessel', 'airway', or 'fissure'. 
    """	
    def __init__(self, ref_particles, test_particles, particle_type=None):
        self._test_particles = test_particles
        self._ref_particles = ref_particles
        self._orientation_vec = None

        self._ref_locator = vtk.vtkPointLocator()
        self._ref_locator.SetDataSet(self._ref_particles)
        self._ref_locator.BuildLocator()

        self._test_locator = vtk.vtkPointLocator()
        self._test_locator.SetDataSet(self._test_particles)
        self._test_locator.BuildLocator()

        self._dist_thresh = None
        self._angle_thresh = None
        self._scale_fraction_thresh = None
        
        if particle_type is not None:
            assert particle_type == 'vessel' or particle_type == 'airway' or \
              particle_type == 'fissure', "Unrecognized particle type"
            if particle_type is 'vessel':
                self._orientation_vec = 'hevec0'
            elif particle_type is 'airway':
                self._orientation_vec = 'hevec2'
            else:
                self._orientation_vec = 'hevec1'
        else:
            raise ValueError('Must specify a particle type')
        
        self._initialize_thresholds()        
        
    def get_particles_dice(self):
        """Computest the Dice coefficient between the ref and test particles
        data sets.

        Returns
        -------
        dice : float
            The Dice coefficient, computed as 2TP/((FP+TP)+(TP+FN)). This is a 
            number between 0 and 1, with 1 indicating perfect agreement and 0
            indicating no agreement.
        """
        TP = 0
        FP = 0
        FN = 0
        
        # Compute TP and FP. Loop over all the test particles, and if a match 
        # is found amoung the ref particles, increment TP. If no matc is 
        # found, increment FP.
        num_test_particles = self._test_particles.GetNumberOfPoints()
        for t in xrange(num_test_particles):
            test_pt = self._test_particles.GetPoint(t)
            test_scale = self._test_particles.GetPointData().\
              GetArray('scale').GetValue(t)

            test_vec = np.zeros(3)
            test_vec[0] = self._test_particles.GetPointData().\
              GetArray(self._orientation_vec).GetComponent(t, 0)
            test_vec[1] = self._test_particles.GetPointData().\
              GetArray(self._orientation_vec).GetComponent(t, 1)
            test_vec[2] = self._test_particles.GetPointData().\
              GetArray(self._orientation_vec).GetComponent(t, 2)

            id_list = vtk.vtkIdList()
            self._ref_locator.FindClosestNPoints(1, test_pt, id_list)
            ref_id = id_list.GetId(0)
            
            dist = LA.norm(np.array(test_pt) - \
                           np.array(self._ref_particles.GetPoint(ref_id)))
                           
            ref_scale = self._ref_particles.GetPointData().\
              GetArray('scale').GetValue(ref_id)
            scale_fraction = np.min([ref_scale, test_scale])/\
              np.max([ref_scale, test_scale])

            ref_vec = np.zeros(3)
            ref_vec[0] = self._ref_particles.GetPointData().\
              GetArray(self._orientation_vec).GetComponent(ref_id, 0)
            ref_vec[1] = self._ref_particles.GetPointData().\
              GetArray(self._orientation_vec).GetComponent(ref_id, 1)
            ref_vec[2] = self._ref_particles.GetPointData().\
              GetArray(self._orientation_vec).GetComponent(ref_id, 2)
            angle = np.arccos(np.clip(np.dot(ref_vec, test_vec)/\
                                LA.norm(ref_vec)/LA.norm(test_vec), -1, 1))

            if dist <= self._dist_thresh and angle <= self._angle_thresh and \
              scale_fraction >= self._scale_fraction_thresh:
                TP += 1
            else:
                FP += 1
                
        # Compute FN
        num_ref_particles = self._ref_particles.GetNumberOfPoints()
        for r in xrange(num_ref_particles):
            ref_pt = self._ref_particles.GetPoint(r)
            ref_scale = self._ref_particles.GetPointData().\
              GetArray('scale').GetValue(r)
            ref_vec = np.zeros(3)              
            ref_vec[0] = self._ref_particles.GetPointData().\
              GetArray(self._orientation_vec).GetComponent(r, 0)
            ref_vec[1] = self._ref_particles.GetPointData().\
              GetArray(self._orientation_vec).GetComponent(r, 1)
            ref_vec[2] = self._ref_particles.GetPointData().\
              GetArray(self._orientation_vec).GetComponent(r, 2)

            id_list = vtk.vtkIdList()
            self._test_locator.FindClosestNPoints(1, ref_pt, id_list)
            test_id = id_list.GetId(0)
            
            dist = LA.norm(np.array(ref_pt) - \
                           np.array(self._test_particles.GetPoint(test_id)))
                           
            test_scale = self._test_particles.GetPointData().\
              GetArray('scale').GetValue(test_id)
            scale_fraction = np.min([ref_scale, test_scale])/\
              np.max([ref_scale, test_scale])

            test_vec = np.zeros(3)
            test_vec[0] = self._test_particles.GetPointData().\
              GetArray(self._orientation_vec).GetComponent(test_id, 0)
            test_vec[1] = self._test_particles.GetPointData().\
              GetArray(self._orientation_vec).GetComponent(test_id, 1)
            test_vec[2] = self._test_particles.GetPointData().\
              GetArray(self._orientation_vec).GetComponent(test_id, 2)
            angle = np.arccos(np.clip(np.dot(ref_vec, test_vec)/\
                                LA.norm(ref_vec)/LA.norm(test_vec), -1, 1))

            if dist > self._dist_thresh or angle > self._angle_thresh or \
              scale_fraction < self._scale_fraction_thresh:
                FN += 1

        dice = 2.*TP/((FP + TP) + (TP + FN))
        return dice
            
    def _initialize_thresholds(self):
        """Use the reference particles data set to figure out the appropriate 
        thresholds for distance, scale, and angle. For each particle in the
        reference data set, the closest neighboring particle is found. 
        Distance, scale, and angle relationships between these neighboring
        particles are used for determining thresholds. After aggregating all
        pairwise relationships, the 99.9th percentile is used as the threshold.
        """
        dists = []
        angles = []
        scale_fractions = []
        
        num_ref_particles = self._ref_particles.GetNumberOfPoints()
        for r in xrange(num_ref_particles):
            ref_pt = self._ref_particles.GetPoint(r)
            ref_scale = self._ref_particles.GetPointData().\
              GetArray('scale').GetValue(r)
            ref_vec = np.zeros(3)              
            ref_vec[0] = self._ref_particles.GetPointData().\
              GetArray(self._orientation_vec).GetComponent(r, 0)
            ref_vec[1] = self._ref_particles.GetPointData().\
              GetArray(self._orientation_vec).GetComponent(r, 1)
            ref_vec[2] = self._ref_particles.GetPointData().\
              GetArray(self._orientation_vec).GetComponent(r, 2)

            id_list = vtk.vtkIdList()
            self._ref_locator.FindClosestNPoints(2, ref_pt, id_list)
            adj_id = id_list.GetId(1)
            
            dist = LA.norm(np.array(ref_pt) - \
                           np.array(self._ref_particles.GetPoint(adj_id)))
            dists.append(dist)
                           
            adj_scale = self._ref_particles.GetPointData().\
              GetArray('scale').GetValue(adj_id)
            scale_fraction = np.min([ref_scale, adj_scale])/\
              np.max([ref_scale, adj_scale])
            scale_fractions.append(scale_fraction)
              
            adj_vec = np.zeros(3)
            adj_vec[0] = self._ref_particles.GetPointData().\
              GetArray(self._orientation_vec).GetComponent(adj_id, 0)
            adj_vec[1] = self._ref_particles.GetPointData().\
              GetArray(self._orientation_vec).GetComponent(adj_id, 1)
            adj_vec[2] = self._ref_particles.GetPointData().\
              GetArray(self._orientation_vec).GetComponent(adj_id, 2)
            angle = np.arccos(np.clip(np.dot(ref_vec, adj_vec)/\
                                      LA.norm(ref_vec)/LA.norm(adj_vec), -1, 1))
            angles.append(angle)
            
        self._dist_thresh = np.percentile(dists, 99.9)
        self._angle_thresh = np.percentile(angles, 99.9)
        self._scale_fraction_thresh = np.percentile(scale_fractions, 0.01)     

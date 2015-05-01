
import subprocess
import numpy as np
from numpy import linalg as LA
from numpy import sum, sqrt
import tempfile, shutil
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import pdb
from cip_python.particles.vessel_particles import VesselParticles

class ParticleMetrics:
    def __init__(self,testing,reference):
        self._testing = testing
        self._reference = reference
        self.point_loc = vtk.vtkPointLocator()
        self.point_loc.SetDataSet(self._testing)
        self.point_loc.BuildLocator()
        self._distance = np.array([])
        self._scale_change = np.array([])
        self._direction_change = np.array([])
    
        self.distance_score = 01
        self.scale_score = 0
        self.direction_score = 0
        self.total_score = 0
    
        self.compute_metrics()
        self.compute_scores()
  
    def compute_metrics(self):
        num_points=self._reference.GetNumberOfPoints()
        idList=vtk.vtkIdList()
        distance = np.zeros(num_points)
        scale_change=np.zeros(num_points)
        direction_change=np.zeros(num_points)
        hh_r=np.zeros(3)
        hh_t=np.zeros(3)
        for ii in xrange(num_points):
            pp_r=self._reference.GetPoint(ii)
            ss_r=self._reference.GetPointData().GetArray('scale').GetValue(ii)
            hh_r[0]=self._reference.GetPointData().GetArray('hevec0').GetComponent(ii,0)
            hh_r[1]=self._reference.GetPointData().GetArray('hevec0').GetComponent(ii,1)
            hh_r[2]=self._reference.GetPointData().GetArray('hevec0').GetComponent(ii,2)
            self.point_loc.FindClosestNPoints(1,pp_r,idList)
            id_t=idList.GetId(0)
            distance[ii]=LA.norm(np.array(pp_r)-np.array(self._testing.GetPoint(id_t)))
            ss_testing = self._testing.GetPointData().GetArray('scale').GetValue(id_t)
            scale_change[ii]=np.abs(ss_r-ss_testing)
            hh_t[0]=self._reference.GetPointData().GetArray('hevec0').GetComponent(id_t,0)
            hh_t[1]=self._reference.GetPointData().GetArray('hevec0').GetComponent(id_t,1)
            hh_t[2]=self._reference.GetPointData().GetArray('hevec0').GetComponent(id_t,2)
            direction_change[ii]=np.arccos(np.abs(sum(hh_r*hh_t)))
            
        self._distance=distance
        self._scale_change=scale_change
        self._direction_change=direction_change
  
    def compute_scores(self):
        inter_distance=self.interparticle_distance(self._reference)
        self.distance_score=np.mean(self._distance/inter_distance)
        ss=vtk_to_numpy(self._reference.GetPointData().GetArray('scale'))
        self.scale_score=np.mean(self._scale_change/ss)
        self.direction_score=np.mean(self._direction_change/(np.pi/2.0))

        self.total_score=1/3*(self.distance_score+self.scale_score+self.direction_score)
    
    def interparticle_distance(self,particles,number_test_points=100):
        pL=vtk.vtkPointLocator()
        pL.SetDataSet(particles)
        pL.BuildLocator()
    
        na=particles.GetNumberOfPoints()
    
        random_ids=np.random.random_integers(0,na-1,number_test_points)
        distance = np.zeros(number_test_points)
        idList=vtk.vtkIdList()
        #Find random closest point and take mean
        for pos,kk in enumerate(random_ids):
            v_p=particles.GetPoint(kk)
            pL.FindClosestNPoints(3,v_p,idList)
            norm1=LA.norm(np.array(v_p)-np.array(particles.GetPoint(idList.GetId(1))))
            norm2=LA.norm(np.array(v_p)-np.array(particles.GetPoint(idList.GetId(2))))
            if (norm1-norm2)/(norm1+norm2) < 0.2:
                distance[pos]=(norm1+norm2)/2
    
        return np.median(distance[distance>0])

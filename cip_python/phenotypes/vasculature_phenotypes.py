#!/usr/bin/python

import vtk, math
import numpy as np
from numpy import linalg as LA
from vtk.util.numpy_support import vtk_to_numpy
from scipy.stats import kde
from scipy.integrate import quadrature
import os.path

import matplotlib
#Use Agg backend to allow non-interactive rendering 
#without relying on X-windows. Other options are: PDF, Cairo..
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from . import Phenotypes
from ..common import ChestConventions
from cip_python.utils.region_type_parser import RegionTypeParser

class VasculaturePhenotypes(Phenotypes):
  """Compute vasculare spectific phenotypes
    
    Paramters
    ---------
    
  """
  def __init__(self,vessel,cid,chest_regions=None,output_prefix=None,min_csa=0,max_csa=90,plot=False):
    #TODO: FIX THIS!! Many wrong parameters, inexistent variables, etc.
    self._chest_region_type_assert(chest_regions,chest_types,pairs)
    self.chest_regions_ = chest_regions
    
    self._vessel = vessel
    self.output_prefix = output_prefix
    self.plot = plot
    self.sigma = 0.18
    self.min_csa = min_csa
    self.max_csa = max_csa
    self.csa_th=np.arange(5,self.max_csa+0.001,5)
    self._spacing = None
    self._sigma0 = 1/np.sqrt(2)/2;
    #Sigma due to the limited pixel resolution (half of 1 pixel)
    self._sigmap = 1/np.sqrt(2)/2
    self._dx = None
    self._number_test_points=1000
    self.factor=0.16
    self._cid=cid
    
    #QC files
    self._qc_suffix = '_vasculaturePhenotypesPlot.png'
    
    Phenotypes.__init__(self)
  
  def declare_pheno_names(self):
  
    """Creates the names of the phenotypes to compute
      
      Returns
      -------
      names : list of strings
      Phenotype names
    """
    
    cols=[]
    cols.append('TBV')
    th = self.csa_th[0]
    cols.append('BV%d'%th)
    for kk in xrange(len(self.csa_th)-1):
      th = self.csa_th[kk]
      th1 = self.csa_th[kk+1]
      cols.append('BV%d_%d'%(th,th1))
    return cols
  
  def get_cid(self):
    """Get the case ID (CID)
    """
    return self._cid
  
  def execute(self):
    
    #Getting info from vtkPolyData with vessel particles
    vessel=self._vessel
    array_v =dict();
            
    for ff in ("scale", "hevec0", "hevec1", "hevec2", "h0", "h1", "h2","ChestRegionChestType"):
      tmp=vessel.GetPointData().GetArray(ff)
      if isinstance(tmp,vtk.vtkDataArray) == False:
        tmp=vessel.GetFieldData().GetArray(ff)
      array_v[ff]=vtk_to_numpy(tmp)
    print "Number of Vessel Points "+str(vessel.GetNumberOfPoints())

    spacing = vtk_to_numpy(vessel.GetFieldData().GetArray("spacing"))
    
    #Get unique value of spacing as the norm 3D vector
    self._spacing=np.sqrt(np.sum(spacing*spacing))
    
    
    #Setting up computation arrays
    csa = np.arange(self.min_csa,self.max_csa,0.05)
    rad = np.sqrt(csa/math.pi)
    bden = np.zeros(csa.size)
    cden = np.zeros(csa.size)

    #Compute interparticle distance
    self._dx = self.interparticle_distance()
  
    print "DX: "+str(self._dx)
    
    tbv=dict()
    bv=dict()
    if self.plot==True:
      profiles=list()
        
    #For each region and type create phenotype table
    c=ChestConventions()
    
    region_type_arr=array_v['ChestRegion']
    
    region_arr = np.zeros(region_type_arr.shape)
    type_arr = np.zeros(region_type_arr.shape)
    
    for kk in xrange(len(region_type_arr)):
      region_arr[kk]=c.GetChestRegionFromValue(region_type_arr[kk])
      type_arr[kk]=c.GetChestTypeFromValue(region_type_arr[kk])
    
    #print c.GetChestTypeValueFromName('Vessel')
    vessel_type = c.GetChestTypeValueFromName('Vessel')
    
    vessel_mask = (type_arr==vessel_type)
    
    if np.sum(vessel_mask==True)==0:
      raise ValueError(\
                       'ChestType does not contain vessels.')

    #Get region information
    if self.chest_regions_ is not None:
      rs=self.chest_regions_
    
    parser = RegionTypeParser(region_arr)
    if rs == None:
      rs = parser.get_all_chest_regions()
    
    if rs is not None:
      for r_id in rs:
        if r_id != 0:
          region_mask = parser.get_mask(chest_region=r_id)
          region_vessel_mask = np.logical_and(region_mask,vessel_mask)
          
          region_name=c.GetChestRegionName(r_id)
          type_name = c.GetChestTypeName(vessel_type_id)
          p_csa=self.compute_bv_profile_from_scale(array_v['scale'][region_vessel_mask])
          n_points = np.sum(region_vessel_mask==True)
          
          if self.plot==True:
            profiles.append([region_name,type_name,p_csa,n_points])

          #Compute blood volume phenotypes integrating along profile
          pheno_name='TBV'
          tbv[r_id]=self.integrate_volume(p_csa,self.min_csa,self.max_csa,n_points,self._dx)
          self.add_pheno([region_name,type_name],pheno_name,tbv[rr_id])
           
          th=self.csa_th[0]
          pheno_name='BV%d'%th
          bv[r_id,th]=self.integrate_volume(p_csa,self.min_csa,th,n_points,self._dx)
          self.add_pheno([region_name,type_name],pheno_name,bv[r_id,th])
                             
          for kk in xrange(len(self.csa_th)-1):
            th = self.csa_th[kk]
            th1 = self.csa_th[kk+1]
            pheno_name='BV%d_%d'%(th,th1)
            bv[rr_id,th]=self.integrate_volume(p_csa,th,th1,n_points,self._dx)
            self.add_pheno([region_name,type_name],pheno_name,bv[rr_id,th])
                             
    #Do quality control plotting for the phenotypes
    if self.plot == True:
      fig=plt.figure()
      n_plots=len(profiles)
      csa=np.arange(self.min_csa,self.max_csa,0.1)
      for ii,values in enumerate(profiles):
        p_csa=values[2]
        n_points=values[3]
        ax1=fig.add_subplot(n_plots,1,ii+1)
        ax1.plot(csa,self._dx*n_points*p_csa.evaluate(csa))
        ax1.grid(True)
        plt.ylabel('Blood Volume (mm^3)')
        plt.xlabel('CSA (mm^2)')
        plt.title('Region '+values[0]+' Type '+values[1])
      fig.savefig(self.output_prefix+self._qc_suffix,dpi=180)
   
    return self._df

  def compute_bv_profile_from_scale(self,scale_arr):
    #Do some automatic bandwithd estimation
    p_csa=kde.gaussian_kde(np.pi*self.vessel_radius_from_sigma(scale_arr)**2)
    return p_csa

  def integrate_volume(self,kernel,min_x,max_x,N,dx):
    intval=quadrature(self._csa_times_pcsa,min_x,max_x,args=[kernel],maxiter=200)
    return dx*N*intval[0]
        
  def csa_times_pcsa(self,p,kernel):
    return p*kernel.evaluate(p)

  def interparticle_distance(self):
    #Create locator
    pL=vtk.vtkPointLocator()
    pL.SetDataSet(self._vessel)
    pL.BuildLocator()

    na=self._vessel.GetNumberOfPoints()

    random_ids=np.random.random_integers(0,na-1,self._number_test_points)
    distance = np.zeros(self._number_test_points)
    idList=vtk.vtkIdList()
    #Find random closest point and take mean
    for pos,kk in enumerate(random_ids):
      v_p=self._vessel.GetPoint(kk)
      pL.FindClosestNPoints(3,v_p,idList)
      norm1=LA.norm(np.array(v_p)-np.array(self._vessel.GetPoint(idList.GetId(1))))
      norm2=LA.norm(np.array(v_p)-np.array(self._vessel.GetPoint(idList.GetId(2))))
      if (norm1-norm2)/(norm1+norm2) < 0.2:
        distance[pos]=(norm1+norm2)/2
      
    return np.median(distance[distance>0])
      
  def vessel_radius_from_sigma(self,scale):
    #return self.spacing*math.sqrt(2)*sigma
    mask = scale< (2/np.sqrt(2)*self._sigma0)
    rad=np.zeros(mask.shape)
    rad[mask]=np.sqrt(2)*(np.sqrt((scale[mask]*self.spacing)**2 + (self._sigma0*self.spacing)**2) -0.5*self._sigma0*self.spacing)
    rad[~mask]=np.sqrt(2)*(np.sqrt((scale[~mask]*self.spacing)**2 + (self._sigmap*self.spacing)**2) -0.5*self._sigmap*self.spacing)
    return rad



from optparse import OptionParser
if __name__ == "__main__":
    desc = """Compute vasculature phenotypes"""
                                    
    parser = OptionParser(description=desc)
    parser.add_option('-v',help='VTK vessel particle filename',
                      dest='vessel_file',metavar='<string>',default=None)
    parser.add_option('-o',help='Output prefix name',
                                    dest='output_prefix',metavar='<string>',default=None)
    parser.add_option('-p', help='Enable plotting', dest='plot', \
                  action='store_true')
    parser.add_option('-c', help='cid', dest='cid',default=None)

    (options,args) =  parser.parse_args()
  
    readerV=vtk.vtkPolyDataReader()
    readerV.SetFileName(options.vessel_file)
    readerV.Update()
    vessel=readerV.GetOutput()

    vp=VasculaturePhenotypes(vessel,options.cid,options.output_prefix,plot=options.plot)
    vp.execute()
                             
    vp.to_csv(os.path.join(output_prefix,'_vasculaturePhenotypes.csv'))

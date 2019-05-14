#!/usr/bin/python

import vtk, math
import numpy as np
from numpy import linalg as LA
from vtk.util.numpy_support import vtk_to_numpy
from scipy.stats import kde
from scipy.integrate import quadrature
import os.path

import matplotlib.pyplot as plt

from cip_python.phenotypes import Phenotypes


class VasculaturePhenotypes(Phenotypes):
  """Compute vasculare spectific phenotypes
    
    Paramters
    ---------
    
  """
  def __init__(self,vessel,cid,output_prefix,spacing=0.625,min_csa=0,max_csa=90,plot=False):
  

    self._vessel = vessel
    self.output_prefix = output_prefix
    self.plot = plot
    self.sigma = 0.18
    self.min_csa = min_csa
    self.max_csa = max_csa
    self.csa_th=np.arange(5,self.max_csa+0.001,5)
    self.spacing = spacing
    self._sigma0 = 1/np.sqrt(2)/2;
    #Sigma due to the limited pixel resolution (half of 1 pixel)
    self._sigmap = 1/np.sqrt(2)/2
    self._dx = None
    self._number_test_points=1000
    self.factor=0.16
    
    Phenotypes.__init__(self,cid)
      
  def declare_quantities(self):
    cols=list()
    cols.append('TBV')
    th = self.csa_th[0]
    cols.append('BV%d'%th)
    for kk in range(len(self.csa_th)-1):
      th = self.csa_th[kk]
      th1 = self.csa_th[kk+1]
      cols.append('BV%d_%d'%(th,th1))
    return cols
  
  def execute(self):
    vessel=self._vessel
    array_v =dict();
            
    for ff in ("scale", "hevec0", "hevec1", "hevec2", "h0", "h1", "h2","ChestRegion","ChestType"):
      tmp=vessel.GetPointData().GetArray(ff)
      if isinstance(tmp,vtk.vtkDataArray) == False:
        tmp=vessel.GetFieldData().GetArray(ff)
      array_v[ff]=vtk_to_numpy(tmp)

    csa = np.arange(self.min_csa,self.max_csa,0.05)
    rad = np.sqrt(csa/math.pi)
    bden = np.zeros(csa.size)
    cden = np.zeros(csa.size)

    print ("Number of Vessel Points "+str(vessel.GetNumberOfPoints()))
    
    #Compute interparticle distance
    self._dx = self.interparticle_distance()
  
    print ("DX: "+str(self._dx))
    
    tbv=dict()
    bv=dict()
    if self.plot==True:
      profiles=list()
        
    #For each region and type create phenotype table
    region_set=set(array_v['ChestRegion'])
    type_id=array_v['ChestType'][0]
        
    for rr_id,rr_name in enumerate(self._region_name):
      region_mask=[]
      if region_set.isdisjoint([rr_id]):
        #Check region hierarchy for this region
        for group in self.region_has(rr_id):
          if region_set.issuperset(group):
            region_mask=array_v['ChestRegion']==group[0]
            for val in group:
              region_mask = np.logical_or(region_mask,array_v['ChestRegion']==val)
      else:
        region_mask= array_v['ChestRegion']==rr_id
      
      if len(region_mask)==0:
        continue

      #Get names for region and type
      region_name = self._region_name[rr_id]
      type_name = self._type_name[type_id]

      #Compute BVx and TBVx for each threshold using quadrature integration
      #Compute p(CSA) based on particle points
      n_points=len(array_v['scale'][region_mask])
      p_csa=kde.gaussian_kde(np.pi*self.vessel_radius_from_sigma(array_v['scale'][region_mask])**2)
      if self.plot==True:
        profiles.append([region_name,type_name,p_csa,n_points])

      tbv[rr_id]=self.integrate_volume(p_csa,self.min_csa,self.max_csa,n_points,self._dx)
      self.add_pheno([region_name,type_name],'TBV',tbv[rr_id])

      th=self.csa_th[0]
      bv[rr_id,th]=self.integrate_volume(p_csa,self.min_csa,th,n_points,self._dx)
      self.add_pheno([region_name,type_name],'BV%d'%th,bv[rr_id,th])
      for kk in range(len(self.csa_th)-1):
        th = self.csa_th[kk]
        th1 = self.csa_th[kk+1]
        bv[rr_id,th]=self.integrate_volume(p_csa,th,th1,n_points,self._dx)
        self.add_pheno([region_name,type_name],'BV%d_%d'%(th,th1),bv[rr_id,th])

    self.save_to_csv(self.output_prefix+'_vasculaturePhenotypes.csv')

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
      fig.savefig(self.output_prefix+'_vasculaturePhenotypesPlot.png',dpi=180)

    

  def integrate_volume(self,kernel,min_x,max_x,N,dx):
    intval=quadrature(self._csa_times_pcsa,min_x,max_x,args=[kernel],maxiter=200)
    return dx*N*intval[0]
        
  def _csa_times_pcsa(self,p,kernel):
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

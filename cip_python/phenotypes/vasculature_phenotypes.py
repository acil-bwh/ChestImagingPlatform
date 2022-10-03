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
from cip_python.common import ChestConventions
from cip_python.utils import RegionTypeParser
from cip_python.classification import kde_bandwidth

class VasculaturePhenotypes(Phenotypes):
    """Compute vasculare spectific phenotypes.
      
      The user can specify chest regions, chest types and region-type pairs over 
      which to compute the phenotypes. The following phenotypes are computed:
      'BV5'
      'BV5_B10'
      'BV10_15'
      'BV15_20'
      'BV20_25'
      'BV25_30'
      'BV30_35'
      'BV35_40'
      'BV40_45'
      'TBV'
      
    """
    def __init__(self,chest_regions=None,chest_types=None, pairs=None,plot=False):
    
        self.min_csa = 0
        self.max_csa = 90
        self.csa_th=np.arange(5,self.max_csa+0.001,5)
        self._spacing = None
        self._sigma0 = 1/np.sqrt(2.)/2.
        #Sigma due to the limited pixel resolution (half of 1 pixel)
        self._sigmap = 1/np.sqrt(2.)/2.
        self._dx = None
        self._number_test_points=5000
        self.factor=0.16
        self.scale_radius_ratio_th = 3
        self.filter_particles_with_scale_radius_ratio=None
        self.old_interarticle_distance=None
        
        #Method to do KDE of prob(CSA)
        self.bw_method='scott'  #options are scott,botev,silverman or a value
        
        #Array name with radius data (optional)
        self.rad_arrayname=None

        self.plot=plot

        self.requested_pheno_names = self.declare_pheno_names()

        c = ChestConventions()

        self.chest_regions_ = None
        if chest_regions is not None:
            tmp = []
            for m in range(0, len(chest_regions)):
                tmp.append(c.GetChestRegionValueFromName(chest_regions[m]))
            self.chest_regions_ = np.array(tmp)

        self.chest_types_ = None
        if chest_types is not None:
            tmp = []
            for m in range(0, len(chest_types)):
                tmp.append(c.GetChestTypeValueFromName(chest_types[m]))
            self.chest_types_ = np.array(tmp)

        self.pairs_ = None
        if pairs is not None:
            self.pairs_ = np.zeros([len(pairs), 2])
            inc = 0
            for p in pairs:
                assert len(p)%2 == 0, \
                    "Specified region-type pairs not understood"
                r = c.GetChestRegionValueFromName(p[0])
                t = c.GetChestTypeValueFromName(p[1])
                self.pairs_[inc, 0] = r
                self.pairs_[inc, 1] = t
                inc += 1

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
        for kk in range(len(self.csa_th)-1):
          th = self.csa_th[kk]
          th1 = self.csa_th[kk+1]
          cols.append('BV%d_%d'%(th,th1))
        return cols
  
    def get_cid(self):
        """Get the case ID (CID)
        """
        return self.cid_
  
    def execute(self,vessel,cid,chest_regions=None,chest_types=None,pairs=None,spacing=np.array([0.625,0.625,0.625])):
        """Compute the phenotypes for the specified structures for the
          specified threshold values.
          
          The following values are computed.
          'TBV':Total blood volume
          'BV5': Blood volume for vessels than 5 mm^2
          'BVx_y: Blood volume for vessels between x mm^2 and y mm^2
          
          Parameters
          ---------
          vessel: vtkPolyDataArray
               vessel particle array
          cid: string
              case ID
          
          chest_regions : array, shape ( R ), optional
          Array of integers, with each element in the interval [0, 255],
          indicating the chest regions over which to compute the LAA. If none
          specified, the chest regions specified in the class constructor
          will be used. If chest regions, chest types, and chest pairs are
          left unspecified both here and in the constructor, then the
          complete set of entities found in the label map will be used.
          
          chest_types : array, shape ( T ), optional
          Array of integers, with each element in the interval [0, 255],
          indicating the chest types over which to compute the LAA. If none
          specified, the chest types specified in the class constructor
          will be used. If chest regions, chest types, and chest pairs are
          left unspecified both here and in the constructor, then the
          complete set of entities found in the label map will be used.
          
          pairs : array, shape ( P, 2 ), optional
          Array of chest-region chest-type pairs over which to compute the
          LAA. The first column indicates the chest region, and the second
          column indicates the chest type. Each element should be in the
          interal [0, 255]. If none specified, the pairs specified in the
          class constructor will be used. If chest regions, chest types, and
          chest pairs are left unspecified both here and in the constructor,
          then the complete set of entities found in the label map will be
          used.
          
          spacing: array, shape (3, ), optional
          Spacing of the volume that generated the particle files. This value is used 
          if the VTK does not contain a FieldData array with spacing information.
          
          Returns
          -------
          df : pandas dataframe
          Dataframe containing info about machine, run time, and chest region
          chest type phenotype quantities.
          fig: matplotlib figure handler
          Figure with plot of blood volume profiles (if plotting has been set to True)
    """


        c = ChestConventions()

        assert type(cid) == str, "cid must be a string"
        self.cid_ = cid

        #Check validity of phenotypes
        phenos_to_compute = self.pheno_names_

        for pheno_name in phenos_to_compute:
            assert pheno_name in self.pheno_names_, \
            "Invalid phenotype name " + pheno_name

        #Getting info from vtkPolyData with vessel particles
        array_v =dict();

        for ff in ("scale", "hevec0", "hevec1", "hevec2", "h0", "h1", "h2","ChestRegionChestType"):
            tmp=vessel.GetPointData().GetArray(ff)
              #if isinstance(tmp,vtk.vtkDataArray) == False:
              #tmp=vessel.GetFieldData().GetArray(ff)
            array_v[ff]=vtk_to_numpy(tmp)
        print ("Number of Vessel Points "+str(vessel.GetNumberOfPoints()))
        
        if self.rad_arrayname is not None:
            tmp=vessel.GetPointData().GetArray(self.rad_arrayname)
            array_v[self.rad_arrayname]=vtk_to_numpy(tmp)
            
            

        #Get unique value of spacing as the norm 3D vector
        if vessel.GetFieldData().GetArray("spacing") == None:
            print ("Spacing information missing in particle vtk file. Setting spacing to (%f,%f,%f)" % (spacing[0],spacing[1],spacing[2]))
            spacing=spacing
        else:
            spacing = vtk_to_numpy(vessel.GetFieldData().GetArray("spacing"))
            print ("Using spacing in particle file: (%f,%f,%f)" % (spacing[0,0],spacing[0,1],spacing[0,2]))

        #Compute single spacing value as the geometric mean of the three spacing values
        self._spacing=np.prod(spacing)**(1/3.0)

        array_v['ChestRegionChestType']=array_v['ChestRegionChestType'].astype('uint16')
        #Check that ChestType contains a relevant vessel type
        type_arr=c.GetChestTypeFromValue((array_v['ChestRegionChestType']))

#        vessel_type = c.GetChestTypeValueFromName('Vessel')
#
#        vessel_mask = (type_arr==vessel_type)
#
#        if np.sum(vessel_mask==True)==0:
#            raise ValueError(\
#                'ChestType does not contain vessels.')

        #Setting up computation arrays
        csa = np.arange(self.min_csa,self.max_csa,0.05)
        rad = np.sqrt(csa/math.pi)
        bden = np.zeros(csa.size)
        cden = np.zeros(csa.size)

        #Compute interparticle distance
        self._dx = self.interparticle_distance(vessel)

        print ("DX: "+str(self._dx))

        profiles=list()

        #For each region and type create phenotype table
        rs = np.array([])
        ts = np.array([])
        ps = np.array([])
        if chest_regions is not None:
            rs = chest_regions
        elif self.chest_regions_ is not None:
            rs = self.chest_regions_
        if chest_types is not None:
            ts = chest_types
        elif self.chest_types_ is not None:
            ts = self.chest_types_
        if pairs is not None:
            ps = pairs
        elif self.pairs_ is not None:
            ps = self.pairs_

        parser = RegionTypeParser(array_v['ChestRegionChestType'])

        if rs.size == 0 and ts.size == 0 and ps.size == 0:
            rs = parser.get_all_chest_regions()
            ts = parser.get_chest_types()
            ps = parser.get_all_pairs()

        #Now compute the phenotpyes and populate the data frame
        for r in rs:
            if r != 0:
                mask_region = parser.get_mask(chest_region=r)
                profile=self.add_pheno_group(array_v, mask_region, mask_region,
                                        None, c.GetChestRegionName(r),
                                        c.GetChestWildCardName())
                profiles.append(profile)

        for t in ts:
            if t != 0:
                mask_type = parser.get_mask(chest_type=t)
                profile=self.add_pheno_group(array_v, mask_type, None, mask_type,
                                         c.GetChestWildCardName(),
                                         c.GetChestTypeName(t))
                profiles.append(profile)
        if ps.size > 0:
            for i in range(0, ps.shape[0]):
                if not (ps[i, 0] == 0 and ps[i, 1] == 0):
                    mask = parser.get_mask(chest_region=int(ps[i, 0]),
                                               chest_type=int(ps[i, 1]))
                    profile=self.add_pheno_group(array_v, mask, None,
                                             None, c.GetChestRegionName(int(ps[i, 0])),
                                             c.GetChestTypeName(int(ps[i, 1])))
                    profiles.append(profile)

        #Do quality control plotting for the phenotypes
        fig=None
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

        return self._df,fig,profiles


    def add_pheno_group(self,array_v,mask,mask_region,mask_type,chest_region_name,chest_type_name):

        mask_sum = np.sum(mask)

        #region_vessel_mask = np.logical_and(mask_region, mask_vessel)
        region_vessel_mask=mask
        
        if mask_sum == 0:
            return None

        if self.rad_arrayname == None:
          vessel_radius = self.vessel_radius_from_sigma(array_v['scale'][region_vessel_mask])
        else:
          vessel_radius = array_v[self.rad_arrayname][region_vessel_mask]
          if self.filter_particles_with_scale_radius_ratio:
            radius_scale = self.vessel_radius_from_sigma(array_v['scale'][region_vessel_mask])
            ratio = np.where(vessel_radius==0, 10, (radius_scale / vessel_radius))
            vessel_radius = np.where(ratio < self.scale_radius_ratio_th, radius_scale ,0)



          

        p_csa = self.compute_bv_profile_from_radius(vessel_radius)
        n_points = np.sum(region_vessel_mask == True)

        #Set out profile set. This output can be used for plotting and additional analysis
        profile=[chest_region_name, chest_type_name, p_csa, n_points, self._dx]
        
        # Compute blood volume phenotypes integrating along profile
        pheno_name = 'TBV'
        tbv = self.integrate_volume(p_csa, self.min_csa, self.max_csa, n_points, self._dx)
        self.add_pheno([chest_region_name, chest_type_name], pheno_name, tbv)

        bv=dict()
        th = self.csa_th[0]
        pheno_name = 'BV%d' % th
        bv[th] = self.integrate_volume(p_csa, self.min_csa, th, n_points, self._dx)
        self.add_pheno([chest_region_name, chest_type_name], pheno_name, bv[th])

        for kk in range(len(self.csa_th) - 1):
            th = self.csa_th[kk]
            th1 = self.csa_th[kk + 1]
            pheno_name = 'BV%d_%d' % (th, th1)
            bv[th] = self.integrate_volume(p_csa, th, th1, n_points, self._dx)
            self.add_pheno([chest_region_name, chest_type_name], pheno_name, bv[th])

        return profile
            
    def botev_kde_bandwidth(self,scale_arr):
        bb=kde_bandwidth.botev_bandwidth()
        return bb.run(scale_arr)
    
    def compute_bv_profile_from_radius(self,radius_arr):
        #Do some automatic bandwithd estimation
        if self.bw_method=='botev':
          bw_value=self.botev_kde_bandwidth(np.pi*radius_arr**2)
          print ("Using botev bw estimation with value=%f"%bw_value)
        else:
          bw_value=self.bw_method
        p_csa=kde.gaussian_kde(np.pi*radius_arr**2,bw_method=bw_value)
        return p_csa

    def integrate_volume(self,kernel,min_x,max_x,N,dx):
        intval=quadrature(self.csa_times_pcsa,min_x,max_x,args=[kernel],maxiter=200)
        return dx*N*intval[0]

    def csa_times_pcsa(self,p,kernel):
        
        return p*kernel[0].evaluate(p)

    def interparticle_distance(self,vessel):
        #Create locator
        pL=vtk.vtkPointLocator()
        pL.SetDataSet(vessel)
        pL.BuildLocator()

        na=vessel.GetNumberOfPoints()

        if self.old_interparticle_distance:
            random_ids=np.random.random_integers(0,na-1,self._number_test_points)
            distance = np.zeros(self._number_test_points)
            idList=vtk.vtkIdList()
        else:
            random_ids=range(0,na)
            distance = np.zeros(na)
            idList=vtk.vtkIdList()

        #Find random closest point and take mean
        for pos,kk in enumerate(random_ids):
          v_p=vessel.GetPoint(kk)
          pL.FindClosestNPoints(3,v_p,idList)
          norm1=LA.norm(np.array(v_p)-np.array(vessel.GetPoint(idList.GetId(1))))
          norm2=LA.norm(np.array(v_p)-np.array(vessel.GetPoint(idList.GetId(2))))
          if (norm1-norm2)/(norm1+norm2) < 0.2:
            distance[pos]=(norm1+norm2)/2.

        return np.median(distance[distance>0])

    def vessel_radius_from_sigma(self,scale):
        #return self.spacing*math.sqrt(2)*sigma
        mask = scale< (2./np.sqrt(2)*self._sigma0)
        rad=np.zeros(mask.shape)
        rad[mask]=np.sqrt(2.)*(np.sqrt((scale[mask]*self._spacing)**2.0 + (self._sigma0*self._spacing)**2.0) -0.5*self._sigma0*self._spacing)
        rad[~mask]=np.sqrt(2.)*(np.sqrt((scale[~mask]*self._spacing)**2.0 + (self._sigmap*self._spacing)**2.0) -0.5*self._sigmap*self._spacing)
        return rad


from optparse import OptionParser
if __name__ == "__main__":
    desc = """Compute vasculature phenotypes"""
                                    
    parser = OptionParser(description=desc)
    parser.add_option('-i',help='VTK vessel particle filename',
                      dest='vessel_file',metavar='<string>',default=None)
    parser.add_option('--out_csv',help='Output csv file in which to store the computed dataframe (ex: cid_vascularePhenotype.csv)',
                                    dest='out_csv',metavar='<string>',default=None)
    parser.add_option('--cid',
                      help='The database case ID', dest='cid',
                      metavar='<string>', default=None)
    parser.add_option('-r',
                      help='Chest regions. Should be specified as a \
                      common-separated list of string values indicating the \
                      desired regions over which to compute the phenotypes. \
                      E.g. LeftLung,RightLung would be a valid input.',
                      dest='chest_regions', metavar='<string>', default=None)
    parser.add_option('-t',
                      help='Chest types. Should be specified as a \
                      common-separated list of string values indicating the \
                      desired types over which to compute the phenotypes. \
                      E.g.: Vessel,NormalParenchyma would be a valid input.',
                      dest='chest_types', metavar='<string>', default=None)
    parser.add_option('-p',
                      help='Chest region-type pairs. Should be \
                      specified as a common-separated list of string values \
                      indicating what region-type pairs over which to compute \
                      phenotypes. For a given pair, the first entry is \
                      interpreted as the chest region, and the second entry \
                      is interpreted as the chest type. E.g. LeftLung,Vessel \
                      would be a valid entry. Note that region-type pairs are \
                      interpreted differently in the case of the TypeFrac \
                      phenotype. In that case, the specified pair is used to \
                      compute the fraction of the chest type within the \
                      chest region.',
                      dest='pairs', metavar='<string>', default=None)
    parser.add_option('--radius_name',
                      help='Array name with the radius information (optional).\
                      If this is not provided the radius will be computed from the scale information.',
                      dest='radius_array_name',metavar='<float>',default=None)
    parser.add_option('--filter_scale',
                      help='Used alongside --radius_name option. Uses scale info to compute phenotypes filtering particles that are \
                      out of ratio compare to dnn sizing  (Flag)',
                      dest='filter_particles_with_scale_radius_ratio',action="store_true")
    parser.add_option('--old_interparticle_distance',
                      help='Calculates interparticle distance from 5000 random particles. This mode is deprecated, now it uses all particles. (Flag)',
                      dest='old_interparticle_distance',action="store_true")
    parser.add_option('-s',
                        help='Spacing of the volume that was used to generate the particles (optional).\
                        This information is used if the spacing field of the particle\'s FieldData is not present.',
                        dest='spacing',metavar='<string>',default=None)
    parser.add_option('--out_plot',help='Output png file with plots of the blood volume profiles (ex: cid_vascularePhenotypePlot.png)',
                                        dest='out_plot',metavar='<string>',default=None)

    (options,args) =  parser.parse_args()


    readerV=vtk.vtkPolyDataReader()
    readerV.SetFileName(options.vessel_file)
    readerV.Update()
    vessel=readerV.GetOutput()

    regions = None
    if options.chest_regions is not None:
        regions = options.chest_regions.split(',')
    types = None
    if options.chest_types is not None:
        types = options.chest_types.split(',')
    pairs = None
    if options.pairs is not None:
        tmp = options.pairs.split(',')
        assert len(tmp)%2 == 0, 'Specified pairs not understood'
        pairs = []
        for i in range(0, int(len(tmp)/2)):
            pairs.append([tmp[2*i], tmp[2*i+1]])

    if options.out_plot is None:
        plot=False
    else:
        plot=True

    if options.spacing is not None:
        spacing=np.array([float(options.spacing),float(options.spacing),float(options.spacing)])
    else:
        spacing = np.array([0.625, 0.625, 0.625])

    vasculature_pheno=VasculaturePhenotypes(chest_regions=regions,chest_types=types,pairs=pairs,plot=plot)
    vasculature_pheno.rad_arrayname=options.radius_array_name
    vasculature_pheno.filter_particles_with_scale_radius_ratio=options.filter_particles_with_scale_radius_ratio
    vasculature_pheno.old_interparticle_distance=options.old_interparticle_distance
    v_df,figure,profiles=vasculature_pheno.execute(vessel,options.cid,spacing=spacing)


    if options.out_csv is not None:
        v_df.to_csv(options.out_csv,index=False)

    if options.out_plot is not None:
        figure.savefig(options.out_plot,dpi=180)


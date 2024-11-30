import numpy as np
from optparse import OptionParser
import warnings
from cip_python.phenotypes import Phenotypes
from cip_python.common import ChestConventions
from cip_python.input_output import ImageReaderWriter
from cip_python.utils import RegionTypeParser
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import sklearn as skl
import pandas as pd

class AirwayPhenotypes(Phenotypes):
    """General purpose class for generating airway-based phenotypes.

    The user can specify chest regions, chest types, or region-type pairs over
    which to compute the phenotypes. Otherwise, the phenotypes will be computed
    as the mean of all the airways for the region/type. The following phenotypes are
    computed using the 'execute' method:
    'innerRadius': mean inner radius (lumen radius) from an fitted elipse in units
    'outerRadius': mean outer radius (outer radius) from an fitted elipse in units
    'wallThickness': mean wall thickness from an fitted elipse in units
    'innerPerimeter': mean inner perimeter in units
    'outerPerimeter': mean outer perimeter in units
    'innerArea': mean inner area in units^2
    'outerArea': mean outer area in units^2
    'wallArea': mean wall area in units^2
    'wallAreaPerc': mean wall area percentage
    'Pi10': regressed wall thickness of an airway with inner perimeter of 10mm
    'wallIntensity': mean wall intensity
    'peakWallIntensity':  mean peak intensity of the wall
    'innerWallIntensity': mean intensity of the inner edge of the airway wall
    'outerWallIntensity': mean intensity of the outer edge of the airway wall
    'power': mean power of the wall
    'numPointsTotal': total number of airway points available
    'numPointsAfterExclusion': effective number of airway points use for the mean comptuation after applying the exclusion criteria
    This quantities can be computed for the following measuring methods currently supported in CIP: FWHM, ZC or PC.

    Parameters
    ----------
    chest_regions : list of strings
        List of chest regions over which to compute the phenotypes.

    chest_types : list of strings
        List of chest types over which to compute the phenotypes.

    pairs : list of lists of strings
        Each element of the list is a two element list indicating a region-type
        pair for which to compute the phenotypes. The first entry indicates the
        chest region of the pair, and second entry indicates the chest type of
        the pair.

    """
    def __init__(self, chest_regions=None, chest_types=None, pairs=None):
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
                
        self.deprecated_phenos_ = []
        
        self.methods_ = ['FWHM','ZC','PC']

        self.dnn_ = True  #Two choices: EdgeDetection or DNN
        
        self.pi_lowerlimit_=7.5
        self.pi_upperlimit_=25.0
        
        Phenotypes.__init__(self)    

    def declare_pheno_names(self):
        """Creates the names of the phenotypes to compute

        Returns
        -------
        names : list of strings
            Phenotype names
        """
        
        #Get phenotypes list from ChestConventions
        c=ChestConventions()
        names = c.AirwayPhenotypeNames
        
        return names

    def get_cid(self):
        """Get the case ID (CID)

        Returns
        -------
        cid : string
            The case ID (CID)
        """
        return self.cid_

    def execute(self, a_pd, cid, chest_regions=None,
                chest_types=None, pairs=None,point_id_exclusion=None):
        """Compute the phenotypes for the specified structures for the
        specified threshold values.

        Parameters
        ----------
        airway_polyData : vtkPolyData with airway points and measurements in the PointData
            vtkPolyData

        cid : string
            Case ID
                        
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
            
         point_id_exclusion: array, shape (E,1), optional
            Array with Point Ids that need to be excluded from the computation \
            (typically the result of a QC assesstment).


        Returns
        -------
        df : pandas dataframe
            Dataframe containing info about machine, run time, and chest region
            chest type phenotype quantities.         
        """
        c = ChestConventions()
        
        assert type(cid) == str, "cid must be a string"
        self.cid_ = cid


        #Derive phenos to compute from list provided
        # As default use the list that is provided in the constructor of phenotypes
        phenos_to_compute = self.pheno_names_
        

        #Warn for phenotpyes that have been deprecated
        for p in phenos_to_compute:
            if p in self.deprecated_phenos_:
                warnings.warn('{} is deprecated. Use TypeFrac instead.'.\
                              format(p), DeprecationWarning)
            
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

        lm = vtk_to_numpy(a_pd.GetPointData().GetArray("ChestRegionChestType")).astype('uint16')
        
        parser = RegionTypeParser(lm)
        
        if rs.size == 0 and ts.size == 0 and ps.size == 0:
            rs = parser.get_all_chest_regions()
            ts = parser.get_chest_types()
            ps = parser.get_all_pairs()

        #Create mask array with exclusions
        keep_mask = np.ones(lm.shape,dtype=bool)
        #We should check that point_id_exclusion is within bounds
        if point_id_exclusion is not None:
            keep_mask[point_id_exclusion]=False
        
        # Now compute the phenotypes and populate the data frame
        for r in rs:
            if r != 0:
                mask_region = parser.get_mask(chest_region=r)
                self.add_pheno_group(a_pd, mask_region, keep_mask, c.GetChestRegionName(r),
                    c.GetChestWildCardName(), phenos_to_compute)
        for t in ts:
            if t != 0:
                mask_type = parser.get_mask(chest_type=t)
                self.add_pheno_group(a_pd, mask_type, keep_mask,
                    c.GetChestWildCardName(),
                    c.GetChestTypeName(t), phenos_to_compute)
        if ps.size > 0:
            for i in range(0, ps.shape[0]):
                if not (ps[i, 0] == 0 and ps[i, 1] == 0):
                    mask = parser.get_mask(chest_region=int(ps[i, 0]),
                                           chest_type=int(ps[i, 1]))
                    self.add_pheno_group(a_pd, mask, keep_mask, c.GetChestRegionName(int(ps[i, 0])),
                        c.GetChestTypeName(int(ps[i, 1])), phenos_to_compute)

        return self._df
    
    def ellipse_perimeter(self,a,b):
        #Use Ramanujan approx https://www.mathsisfun.com/geometry/ellipse-perimeter.html
        h=(a-b)**2/(a+b)**2
        p = np.pi * (a+b)*(1+(3*h/(10+np.sqrt(4-3*h))))
        return p
    
    def mask_from_metrics(self,mean_metrics,ellip_metrics):
        num_points= mean_metrics.shape[0]
        metrics_mask = np.zeros((num_points,),dtype=bool)
        #Apply the trivial solution for now.
        #We could implement problem specific thresholds
        metrics_mask[:]=True

        return metrics_mask
    def add_pheno_group(self, a_pd, mask, keep_mask, chest_region,
                        chest_type, phenos_to_compute):
        """This function computes phenotypes and adds them to the dataframe with
        the 'add_pheno' method. 
        
        Parameters
        ----------
        a_pd : airway vtkPolyData with measurements
            vtkPolyData

        mask : boolean array, shape ( X, Y, Z ), optional
            Boolean mask where True values indicate presence of the structure
            of interest
            
        keep_mask: boolean array, shape ( X, Y, Z ), optional
            Boolean mask for points that passed QC (provided by an external list)

        chest_region : string
            Name of the chest region in the (region, type) key used to populate
            the dataframe

        chest_type : string
            Name of the chest region in the (region, type) key used to populate
            the dataframe

        phenos_to_compute : list of strings
            Names of the phenotype used to populate the dataframe

        """

        if a_pd is not None:
            #Get arrays to do measurements
            if self.dnn_ == True:
                self.add_pheno_group_dnn(a_pd,mask, keep_mask, chest_region,
                        chest_type, phenos_to_compute)
            else:
                self.add_pheno_group_ellipse(a_pd,mask, keep_mask, chest_region,
                        chest_type, phenos_to_compute)


    def add_pheno_group_ellipse(self,a_pd, mask, keep_mask, chest_region,
                        chest_type, phenos_to_compute):
        """This function computes airway phenotypes using point data array corresponding
           to traditional edge detection methods with ellipse fitting. The particle dataset relies
            in two point data arrays: airwaymetrics-[method]-ellipse and airwaymetrics-[method]-mean.

           Parameters
           ----------
           a_pd : airway vtkPolyData with measurements
               vtkPolyData

           mask : boolean array, shape ( X, Y, Z ), optional
               Boolean mask where True values indicate presence of the structure
               of interest

           keep_mask: boolean array, shape ( X, Y, Z ), optional
               Boolean mask for points that passed QC (provided by an external list)

           chest_region : string
               Name of the chest region in the (region, type) key used to populate
               the dataframe

           chest_type : string
               Name of the chest region in the (region, type) key used to populate
               the dataframe

           phenos_to_compute : list of strings
               Names of the phenotype used to populate the dataframe

           """

        num_points_total = np.sum(mask)

        for mm in self.methods_:
            #Check method is avail
            arrEllipseName='airwaymetrics-%s-ellipse'%mm
            arrMeanName='airwaymetrics-%s-mean'%mm
            if a_pd.GetPointData().GetArray(arrEllipseName) == None or \
                a_pd.GetPointData().GetArray(arrMeanName) == None:
                continue

            ellip_metrics=vtk_to_numpy(a_pd.GetPointData().GetArray(arrEllipseName))
            mean_metrics=vtk_to_numpy(a_pd.GetPointData().GetArray(arrMeanName))

            #Select region/type requested based on mask
            ellip_metrics=ellip_metrics[(mask & keep_mask),:]
            mean_metrics=mean_metrics[(mask & keep_mask),:]

            #Define additional points of exclusion based on internal metrics
            metrics_mask=self.mask_from_metrics(mean_metrics,ellip_metrics)

            #Refine pheno arrays with mask from metrics
            ellip_metrics=ellip_metrics[metrics_mask,:]
            mean_metrics=mean_metrics[metrics_mask,:]
            mask_sum = np.sum((metrics_mask))

            if mask_sum==0:
                continue

            ai = ellip_metrics[:,1]
            bi = ellip_metrics[:,0]
            ao = ellip_metrics[:,4]
            bo = ellip_metrics[:,3]

            #Setting up regressor for Pi10 and Pi15 metrics
            sqrtwa = np.sqrt(np.pi *( ao * bo - ai * bi))
            peri = self.ellipse_perimeter(ai,bi)
            #limit pi for pi15
            mask_peri = ((peri>=self.pi_lowerlimit_) & (peri<=self.pi_upperlimit_))
            regr = skl.linear_model.LinearRegression()
            peri=peri.reshape(-1,1)
            regr.fit(peri[mask_peri], sqrtwa[mask_peri])

            for pheno_name in phenos_to_compute:
                assert pheno_name in self.pheno_names_, \
                "Invalid phenotype name " + pheno_name
                pheno_val = None
                if pheno_name == 'innerRadius':
                    pheno_val = np.mean(np.sqrt(ai*bi))
                elif pheno_name == 'outerRadius':
                    pheno_val = np.mean(np.sqrt(ao*bo))
                elif pheno_name == 'wallThickness':
                    #Wall thickness from two concentric ellipses can be computed
                    # wt = (ao*bo - ai*bi)/(sqrt(ao*bo)+sqrt(ai*bi))
                    pheno_val = np.mean((ao*bo-ai*bi)/(np.sqrt(ai*bi)+np.sqrt(ao*bo)))
                elif pheno_name == 'innerPerimeter':
                    pheno_val = np.mean(self.ellipse_perimeter(ai,bi))
                elif pheno_name == 'outerPerimeter':
                    pheno_val = np.mean(self.ellipse_perimeter(ao,bo))
                elif pheno_name == 'innerArea':
                    pheno_val = np.mean( np.pi * ai * bi)
                elif pheno_name == 'outerArea':
                    pheno_val = np.mean( np.pi * ao * bo)
                elif pheno_name == 'wallArea':
                    pheno_val = np.mean(np.pi *( ao * bo - ai * bi) )
                elif pheno_name == 'wallAreaPerc':
                    pheno_val = np.mean(100.0*(ao * bo -  ai * bi)/(ao*bo))
                elif pheno_name == 'Pi10':
                    pheno_val = regr.predict([[10]])
                elif pheno_name == 'Pi15':
                    pheno_val = regr.predict([[15]])
                elif pheno_name == 'wallIntensity':
                    pheno_val = np.mean(mean_metrics[:,3])
                elif pheno_name == 'peakWallIntensity':
                    pheno_val = np.mean(mean_metrics[:,9])
                elif pheno_name == 'innerWallIntensity':
                    pheno_val = np.mean(mean_metrics[:,10])
                elif pheno_name == 'outerWallIntensity':
                    pheno_val = np.mean(mean_metrics[:,11])
                elif pheno_name == 'power':
                    pheno_val = np.mean(mean_metrics[:,20])
                elif pheno_name == 'numPointsTotal':
                    pheno_val = num_points_total
                elif pheno_name == 'numPointsAfterExclusion':
                    pheno_val = mask_sum

                if pheno_val is not None:
                    # self.add_pheno([chest_region, chest_type, mm],
                    #               pheno_name, pheno_val)
                    self.add_pheno([chest_region, chest_type],
                                   pheno_name, pheno_val)


    def add_pheno_group_dnn(self, a_pd, mask, keep_mask, chest_region,
                        chest_type, phenos_to_compute):
        """This function computes airway phenotypes using point data array corresponding
              to a dnn sizing method. The particle dataset relies
               in two point data arrays: dnn_lumen_radius and dnn_wall_thickness.

              Parameters
              ----------
              a_pd : airway vtkPolyData with measurements
                  vtkPolyData

              mask : boolean array, shape ( X, Y, Z ), optional
                  Boolean mask where True values indicate presence of the structure
                  of interest

              keep_mask: boolean array, shape ( X, Y, Z ), optional
                  Boolean mask for points that passed QC (provided by an external list)

              chest_region : string
                  Name of the chest region in the (region, type) key used to populate
                  the dataframe

              chest_type : string
                  Name of the chest region in the (region, type) key used to populate
                  the dataframe

              phenos_to_compute : list of strings
                  Names of the phenotype used to populate the dataframe

              """
        num_points_total = np.sum(mask)

        # Check method is avail
        arrLumenRadiusName = 'dnn_lumen_radius'
        arrWallName = 'dnn_wall_thickness'
        if a_pd.GetPointData().GetArray(arrLumenRadiusName) == None or \
                a_pd.GetPointData().GetArray(arrWallName) == None:
            return

        lr_metrics = vtk_to_numpy(a_pd.GetPointData().GetArray(arrLumenRadiusName))
        wt_metrics = vtk_to_numpy(a_pd.GetPointData().GetArray(arrWallName))

        print (lr_metrics.shape)
        # Select region/type requested based on mask
        lr_metrics = lr_metrics[(mask & keep_mask)]
        wt_metrics = wt_metrics[(mask & keep_mask)]

        # Define additional points of exclusion based on internal metrics
        #metrics_mask = self.mask_from_metrics(mean_metrics, ellip_metrics)

        # Refine pheno arrays with mask from metrics
        #lr_metrics = lr_metrics[metrics_mask, :]
        #wt_metrics = wt_metrics[metrics_mask, :]

        mask_sum = np.sum(mask)

        if mask_sum == 0:
            return

        # Setting up regressor for Pi10 and Pi15 metrics
        sqrtwa = np.sqrt(np.pi * ( (lr_metrics+wt_metrics)**2 - (lr_metrics**2) ))
        peri = 2.0*np.pi*lr_metrics

        # limit pi for pi15
        mask_peri = ((peri >= self.pi_lowerlimit_) & (peri <= self.pi_upperlimit_))
        regr = skl.linear_model.LinearRegression()
        peri = peri.reshape(-1, 1)
        regr.fit(peri[mask_peri], sqrtwa[mask_peri])

        for pheno_name in phenos_to_compute:
            assert pheno_name in self.pheno_names_, \
                "Invalid phenotype name " + pheno_name
            pheno_val = None
            if pheno_name == 'innerRadius':
                pheno_val = np.mean(lr_metrics)
            elif pheno_name == 'outerRadius':
                pheno_val = np.mean(lr_metrics+wt_metrics)
            elif pheno_name == 'wallThickness':
                # Wall thickness from two concentric ellipses can be computed
                # wt = (ao*bo - ai*bi)/(sqrt(ao*bo)+sqrt(ai*bi))
                pheno_val = np.mean(wt_metrics)
            elif pheno_name == 'innerPerimeter':
                pheno_val = np.mean(peri)
            elif pheno_name == 'outerPerimeter':
                pheno_val = np.mean(2.0*np.pi*(lr_metrics+wt_metrics))
            elif pheno_name == 'innerArea':
                pheno_val = np.mean(np.pi * lr_metrics**2)
            elif pheno_name == 'outerArea':
                pheno_val = np.mean(np.pi * (lr_metrics+wt_metrics)**2)
            elif pheno_name == 'wallArea':
                pheno_val = np.mean(np.pi * ( (lr_metrics+wt_metrics)**2 - (lr_metrics**2) ))
            elif pheno_name == 'wallAreaPerc':
                pheno_val = np.mean(100.0 * ((lr_metrics+wt_metrics)**2 - (lr_metrics**2)) / (lr_metrics+wt_metrics)**2)
            elif pheno_name == 'Pi10':
                pheno_val = regr.predict([[10]])
            elif pheno_name == 'Pi15':
                pheno_val = regr.predict([[15]])
            elif pheno_name == 'wallIntensity':
                pheno_val = None
            elif pheno_name == 'peakWallIntensity':
                pheno_val = None
            elif pheno_name == 'innerWallIntensity':
                pheno_val = None
            elif pheno_name == 'outerWallIntensity':
                pheno_val = None
            elif pheno_name == 'power':
                pheno_val = None
            elif pheno_name == 'numPointsTotal':
                pheno_val = num_points_total
            elif pheno_name == 'numPointsAfterExclusion':
                pheno_val = mask_sum

            if pheno_val is not None:
                # self.add_pheno([chest_region, chest_type, mm],
                #               pheno_name, pheno_val)
                self.add_pheno([chest_region, chest_type],
                               pheno_name, pheno_val)


if __name__ == "__main__" and __package__ is None:
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    
    desc = """Generates airway phenotypes from a airway particles vtk file \
    with airway measurements. Note that in general, specified chest-regions, chest-types, and \
    region-type pairs indicate the parts of the airway particle over which the requested \
    phenotypes are computed based on the ChestRegionType point data array."""
    
    parser = OptionParser(description=desc)
    parser.add_option('--in_pd',
                      help='Input airway particles', dest='in_pd', metavar='FILE',
                      default=None)
    parser.add_option('--out_csv',
                      help='Output csv file in which to store the computed \
                      dataframe', dest='out_csv', metavar='FILE',
                      default=None)
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
                      
    parser.add_option('-e',
                      help='List of exclusion airway particles that should not be \
                      considered for computation. The supported format is a text file with point ids',
                      dest='exclusion_list', metavar='<string>',default=None)

    parser.add_option('--pi_ll',
                      help='Lower limit of the internal airway perimeter to compute Pi-related measurements',
                           dest='pi_ll',type='float',metavar='<float>',default=7.5)
    parser.add_option('--pi_ul',
                      help='Upper limit of the internal airway perimeter to compute Pi-related measurments',
                           dest='pi_ul',type='float',metavar='<float>',default=25)

    parser.add_option('--dnn',
                      help='Use dnn measurements',
                      action="store_true", dest="dnn",
                      )

    (options, args) = parser.parse_args()
    
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(options.in_pd)
    reader.Update()
    
    airway_particles = reader.GetOutput()

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
        for i in range(0, len(tmp)/2):
            pairs.append([tmp[2*i], tmp[2*i+1]])

    exclusion_list = None
    if options.exclusion_list is not None:
        el_df=pd.read_csv(options.exclusion_list,header=None)
        assert len(el_df.columns) == 1, 'Unknown format for exclusion list'
        exclusion_list = el_df[0].values

    airway_pheno = AirwayPhenotypes(chest_regions=regions,
            chest_types=types, pairs=pairs)

    airway_pheno.pi_lowerlimit_=float(options.pi_ll)
    airway_pheno.pi_upperlimit_=float(options.pi_ul)

    airway_pheno.dnn_=options.dnn

    df = airway_pheno.execute(airway_particles, options.cid,point_id_exclusion=exclusion_list)

    if options.out_csv is not None:
        df.to_csv(options.out_csv, index=False)


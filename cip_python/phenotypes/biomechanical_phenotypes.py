import numpy as np
from scipy.stats import mode, kurtosis, skew
from optparse import OptionParser
import warnings
from cip_python.phenotypes import Phenotypes
from cip_python.common import ChestConventions
from cip_python.input_output import ImageReaderWriter
from cip_python.utils import RegionTypeParser

class BiomechanicalPhenotypes(Phenotypes):
    """General purpose class for generating strain-based phenotypes.

    The user can specify chest regions, chest types, or region-type pairs over
    which to compute the phenotypes. Otherwise, the phenotypes will be computed
    over all structures in the specified labelmap. The following phenotypes are
    computed using the 'execute' method:
    'JMean': Mean value of the Jacobian values
    'JMean_Minus1': Mean value of the  Jacobian values only considering <1
    'JMean_Major1': Mean value of the Jacobian values only considering >1
    'JStd': Standard deviation of the Jacobian values
    'JKurtosis': Kurtosis of the Jacobian values. Fisher's definition is
    used, meaning that normal distribution has kurtosis of 0. The calculation
    is corrected for statistical bias.
    'JSkewness': Skewness of the jacobian values. The calculation is
    corrected for statistical bias.
    'JMode': Mode of the Jacobian values
    'JMedian': Median of the Jacobian values
    'JMin': Min Jacobian value for the structure
    'JMax': Max Jacobian value for the structure
    'ADIMean': Mean value of the ADI values
    'ADIStd': Standard deviation of the ADI values
    'ADIKurtosis': Kurtosis of the ADI values. Fisher's definition is
    used, meaning that normal distribution has kurtosis of 0. The calculation
    is corrected for statistical bias.
    'ADISkewness': Skewness of the ADI values. The calculation is
    corrected for statistical bias.
    'ADIMode': Mode of the ADI values
    'ADIMedian': Median of the ADI values
    'ADIMin': Min ADI value for the structure
    'ADIMax': Max ADI value for the structure
    'SRIMean': Mean value of the SRI values
    'SRIStd': Standard deviation of the SRI values
    'SRIKurtosis': Kurtosis of the SRI values. Fisher's definition is
    used, meaning that normal distribution has kurtosis of 0. The calculation
    is corrected for statistical bias.
    'SRISkewness': Skewness of the SRI values. The calculation is
    corrected for statistical bias.
    'SRIMode': Mode of the SRI values
    'SRIMedian': Median of the SRI values
    'SRIMin': Min SRI value for the structure
    'SRIMax': Max SRI value for the structure

    
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

    pheno_names : list of strings, optional
        Names of phenotypes to compute. These names must conform to the
        accepted phenotype names, listed above. If none are given, all
        will be computed.
    """
    def __init__(self, chest_regions=None, chest_types=None, pairs=None,
                 pheno_names=None):
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
                
        self.requested_pheno_names = pheno_names
        
        self.deprecated_phenos_ = ['NormalParenchyma', 'PanlobularEmphysema',
                                   'ParaseptalEmphysema', 'MildCentrilobularEmphysema',
                                   'ModerateCentrilobularEmphysema', 'SevereCentrilobularEmphysema',
                                   'MildParaseptalEmphysema']

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
        names = c.BiomechanicalPhenotypeNames
        
        return names

    def get_cid(self):
        """Get the case ID (CID)

        Returns
        -------
        cid : string
            The case ID (CID)
        """
        return self.cid_

    def execute(self, lm, cid, spacing, J=None, ADI=None, SRI=None,  chest_regions=None,
                chest_types=None, pairs=None, pheno_names=None):
        """Compute the phenotypes for the specified structures for the
        specified threshold values.

        The following values are computed for the specified structures.
	'JMean': Mean value of the Jacobian values
    	'JMean_Minus1': Mean value of the  Jacobian values only considering <1
    	'JMean_Major1': Mean value of the Jacobian values only considering >1
	'JStd': Standard deviation of the Jacobian values
    	'JKurtosis': Kurtosis of the Jacobian values. Fisher's definition is
    	used, meaning that normal distribution has kurtosis of 0. The calculation
    	is corrected for statistical bias.
    	'JSkewness': Skewness of the jacobian values. The calculation is
    	corrected for statistical bias.
    	'JMode': Mode of the Jacobian values
    	'JMedian': Median of the Jacobian values
    	'JMin': Min Jacobian value for the structure
    	'JMax': Max Jacobian value for the structure
    	'ADIMean': Mean value of the ADI values
    	'ADIStd': Standard deviation of the ADI values
    	'ADIKurtosis': Kurtosis of the ADI values. Fisher's definition is
    	used, meaning that normal distribution has kurtosis of 0. The calculation
    	is corrected for statistical bias.
    	'ADISkewness': Skewness of the ADI values. The calculation is
    	corrected for statistical bias.
    	'ADIMode': Mode of the ADI values
    	'ADIMedian': Median of the ADI values
    	'ADIMin': Min ADI value for the structure
    	'ADIMax': Max ADI value for the structure
    	'SRIMean': Mean value of the SRI values
    	'SRIStd': Standard deviation of the SRI values
    	'SRIKurtosis': Kurtosis of the SRI values. Fisher's definition is
    	used, meaning that normal distribution has kurtosis of 0. The calculation
    	is corrected for statistical bias.
    	'SRISkewness': Skewness of the SRI values. The calculation is
    	corrected for statistical bias.
    	'SRIMode': Mode of the SRI values
    	'SRIMedian': Median of the SRI values
    	'SRIMin': Min SRI value for the structure
    	'SRIMax': Max SRI value for the structure		        
        
	Parameter
        ----------
        lm : array, shape ( X, Y, Z )
            The 3D label map array

        cid : string
            Case ID

        spacing : array, shape ( 3 )
            The x, y, and z spacing, respectively, of the CT volume

        J : array, shape ( X, Y, Z )
            The 3D J image array

        ADI : array, shape ( X, Y, Z )
            The 3D ADI image array

	SRI : array, shape ( X, Y, Z )
            The 3D SRI image array
                
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

        pheno_names : list of strings, optional
            Names of phenotypes to compute. These names must conform to the
            accepted phenotype names, listed above. If none are given, all
            will be computed. Specified names given here will take precedence
            over any specified in the constructor.

        Returns
        -------
        df : pandas dataframe
            Dataframe containing info about machine, run time, and chest region
            chest type phenotype quantities.         
        """
        c = ChestConventions()
        if J is not None:
            assert len(ct.shape) == len(lm.shape), \
              "CT and label map are not the same dimension"    

        dim = len(lm.shape)
        if J is not None:
            for i in range(0, dim):
                assert ct.shape[0] == lm.shape[0], \
                  "Disagreement in J and label map dimension"

        if ADI is not None:
            assert len(ADI.shape) == len(lm.shape), \
              "ADI and label map are not the same dimension"

        dim = len(lm.shape)
        if ADI is not None:
            for i in range(0, dim):
                assert ADI.shape[0] == lm.shape[0], \
                  "Disagreement in ADI and label map dimension"
    
        if SRI is not None:
            assert len(SRI.shape) == len(lm.shape), \
              "SRI and label map are not the same dimension"

        dim = len(lm.shape)
        if SRI is not None:
            for i in range(0, dim):
                assert SRI.shape[0] == lm.shape[0], \
                  "Disagreement in SRI and label map dimension"


        assert type(cid) == str, "cid must be a string"
        self.cid_ = cid
        self._spacing = spacing


        #Derive phenos to compute from list provided
        # As default use the list that is provided in the constructor of phenotypes
        phenos_to_compute = self.pheno_names_
        
        if pheno_names is not None:
            phenos_to_compute = pheno_names
        elif self.requested_pheno_names is not None:
            phenos_to_compute = self.requested_pheno_names

        #Deal with wildcard: defaul to main pheno list if wildcard is provided
        if c.GetChestWildCardName() in phenos_to_compute:
            phenos_to_compute = self.pheno_names_


        #Check validity of phenotypes
        for pheno_name in phenos_to_compute:
            assert pheno_name in self.pheno_names_, \
                  "Invalid phenotype name " + pheno_name

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

        parser = RegionTypeParser(lm)
        
        if rs.size == 0 and ts.size == 0 and ps.size == 0:
            rs = parser.get_all_chest_regions()
            ts = parser.get_chest_types()
            ps = parser.get_all_pairs()

        # Now compute the phenotypes and populate the data frame
        for r in rs:
            if r != 0:
                mask_region = parser.get_mask(chest_region=r)
                self.add_pheno_group(ct, mask_region, mask_region,
                    None, c.GetChestRegionName(r),
                    c.GetChestWildCardName(), phenos_to_compute)
        for t in ts:
            if t != 0:
                mask_type = parser.get_mask(chest_type=t)
                self.add_pheno_group(ct, mask_type, None, mask_type,
                    c.GetChestWildCardName(),
                    c.GetChestTypeName(t), phenos_to_compute)
        if ps.size > 0:
            for i in range(0, ps.shape[0]):
                if not (ps[i, 0] == 0 and ps[i, 1] == 0):
                    mask = parser.get_mask(chest_region=int(ps[i, 0]),
                                           chest_type=int(ps[i, 1]))
                    if 'TypeFrac' in phenos_to_compute:
                        mask_region = \
                          parser.get_mask(chest_region=int(ps[i, 0]))
                    else:
                        mask_region = None

                    self.add_pheno_group(ct, mask, mask_region,
                        None, c.GetChestRegionName(int(ps[i, 0])),
                        c.GetChestTypeName(int(ps[i, 1])), phenos_to_compute)

        return self._df

    def add_pheno_group(self, J, ADI, SRI, mask, mask_region, mask_type, chest_region, 
                        chest_type, phenos_to_compute):
        """This function computes phenotypes and adds them to the dataframe with
        the 'add_pheno' method. 
        
        Parameters
        ----------
        J : array, shape ( X, Y, Z )
            The 3D CT image array

	ADI : array, shape ( X, Y, Z )
            The 3D CT image array

	SRI : array, shape ( X, Y, Z )
            The 3D CT image array

        mask : boolean array, shape ( X, Y, Z ), optional
            Boolean mask where True values indicate presence of the structure
            of interest
            
        mask_region: boolean array, shape ( X, Y, Z ), optional
            Boolean mask for the corresponding region
            
        mask_type: boolean array, shape ( X, Y, Z ), optional
            Boolean mask for the corresponding type

        chest_region : string
            Name of the chest region in the (region, type) key used to populate
            the dataframe

        chest_type : string
            Name of the chest region in the (region, type) key used to populate
            the dataframe

        phenos_to_compute : list of strings
            Names of the phenotype used to populate the dataframe

        References
        ----------
        1. J Biomech. Three dimensional characterization of regional lung deformation
        """
        mask_sum = np.sum(mask)
        print ("Computing phenos for region %s and type %s"%(chest_region,chest_type))
        if J is not None:
            J_mask=J[mask]
            hus=dict()
            hus[extended] = J[np.logical_and(mask, J < 1)]
            hus[reduced] = J[np.logical_and(mask, J > 1)]
	    
            ADI_mask=ADI[mask]
            SRI_mask=SRI[mask]
	
            for pheno_name in phenos_to_compute:
                assert pheno_name in self.pheno_names_, \
                  "Invalid phenotype name " + pheno_name
                pheno_val = None
                if pheno_name == 'JMean' and mask_sum > 0:
                    pheno_val = np.mean(J_mask)
                elif pheno_name == 'JStd' and mask_sum > 0:
                    pheno_val = np.std(J_mask)
                elif pheno_name == 'JKurtosis' and mask_sum > 0:
                    pheno_val = kurtosis(J_mask, bias=False, fisher=True)
                elif pheno_name == 'JSkewness' and mask_sum > 0:
                    pheno_val = skew(J_mask, bias=False)
                elif pheno_name == 'JMode' and mask_sum > 0:
                    min_val = np.min(J_mask)
                    pheno_val = np.argmax(np.bincount(J_mask + \
                        np.abs(min_val))) - np.abs(min_val)
                elif pheno_name == 'JMedian' and mask_sum > 0:
                    pheno_val = np.median(J_mask)
                elif pheno_name == 'JMin' and mask_sum > 0:
                    pheno_val = np.min(J_mask)
                elif pheno_name == 'JMax' and mask_sum > 0:
                    pheno_val = np.max(J_mask)
                elif pheno_name == 'JMean_Minus1' and mask_sum > 0:
                    if hus[reduced].shape[0] > 0:
                        pheno_val = np.mean(hus[reduced])
                elif pheno_name == 'JMean_Major1' and mask_sum > 0:
                    if hus[extended].shape[0] > 0:
                        pheno_val = np.mean(hus[extended])
                elif pheno_name == 'ADIMean' and mask_sum > 0:
                    pheno_val = np.mean(ADI_mask)
                elif pheno_name == 'ADIStd' and mask_sum > 0:
                    pheno_val = np.std(ADI_mask)
                elif pheno_name == 'ADIKurtosis' and mask_sum > 0:
                    pheno_val = kurtosis(ADI_mask, bias=False, fisher=True)
                elif pheno_name == 'ADISkewness' and mask_sum > 0:
                    pheno_val = skew(ADI_mask, bias=False)
                elif pheno_name == 'ADIMode' and mask_sum > 0:
                    min_val = np.min(ADI_mask)
                    pheno_val = np.argmax(np.bincount(ADI_mask + \
                        np.abs(min_val))) - np.abs(min_val)
                elif pheno_name == 'ADIMedian' and mask_sum > 0:
                    pheno_val = np.median(ADI_mask)
                elif pheno_name == 'ADIMin' and mask_sum > 0:
                    pheno_val = np.min(ADI_mask)
                elif pheno_name == 'ADIMax' and mask_sum > 0:
                    pheno_val = np.max(ADI_mask)
                elif pheno_name == 'SRIMean' and mask_sum > 0:
                    pheno_val = np.mean(SRI_mask)
                elif pheno_name == 'SRIStd' and mask_sum > 0:
                    pheno_val = np.std(SRI_mask)
                elif pheno_name == 'SRIKurtosis' and mask_sum > 0:
                    pheno_val = kurtosis(SRI_mask, bias=False, fisher=True)
                elif pheno_name == 'SRISkewness' and mask_sum > 0:
                    pheno_val = skew(SRI_mask, bias=False)
                elif pheno_name == 'SRIMode' and mask_sum > 0:
                    min_val = np.min(SRI_mask)
                    pheno_val = np.argmax(np.bincount(SRI_mask + \
                        np.abs(min_val))) - np.abs(min_val)
                elif pheno_name == 'SRIMedian' and mask_sum > 0:
                    pheno_val = np.median(SRI_mask)
                elif pheno_name == 'SRIMin' and mask_sum > 0:
                    pheno_val = np.min(SRI_mask)
                elif pheno_name == 'SRIMax' and mask_sum > 0:
                    pheno_val = np.max(SRI_mask)


                if pheno_val is not None:
                    self.add_pheno([chest_region, chest_type],
                                   pheno_name, pheno_val)
                    
        if 'TypeFrac' in phenos_to_compute and mask is not None and \
            mask_region is not None:
            denom = np.sum(mask_region)
            if denom > 0:
                pheno_val = float(np.sum(mask))/float(denom)
                self.add_pheno([chest_region, chest_type],
                                'TypeFrac', pheno_val)

if __name__ == "__main__":
    desc = """Generates parenchyma phenotypes given input CT and segmentation \
    data. Note that in general, specified chest-regions, chest-types, and \
    region-type pairs indicate the parts of the image over which the requested \
    phenotypes are computed. An exception is the TypeFrac phenotype; in that 
    case, a region-type pair must be specified. The TypeFrac phenotype is \
    computed by calculating the fraction of the pair's chest-region that is \
    covered by the pair's chest-type. For example, if a user is interested in \
    the fraction of paraseptal emphysema in the left lung, he/she would \
    specify the TypeFrac phenotype and the LeftLung,ParaseptalEmphysema \
    region-type pair."""
    
    parser = OptionParser(description=desc)
    parser.add_option('--in_J',
                      help='Input J file', dest='in_J', metavar='<string>',
                      default=None)
    parser.add_option('--in_ADI',
                      help='Input ADI file', dest='in_ADI', metavar='<string>',
                      default=None)
    parser.add_option('--in_SRI',
                      help='Input SRI file', dest='in_SRI', metavar='<string>',
                      default=None)
    parser.add_option('--in_lm',
                      help='Input label map containing structures of interest',
                      dest='in_lm', metavar='<string>', default=None)
    parser.add_option('--out_csv',
                      help='Output csv file in which to store the computed \
                      dataframe', dest='out_csv', metavar='<string>',
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

    (options, args) = parser.parse_args()

    image_io = ImageReaderWriter()
    lm, lm_header = image_io.read_in_numpy(options.in_lm)

    J = None
    ADI = None
    SRI = None	
    if options.in_J is not None:
        J, J_header = image_io.read_in_numpy(options.in_J)    
    
    if options.in_ADI is not None:
        ADI, ADI_header = image_io.read_in_numpy(options.in_ADI)

    if options.in_SRI is not None:
        SRI, SRI_header = image_io.read_in_numpy(options.in_SRI)

    spacing = lm_header['spacing']


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

    biomech_pheno = BiomechanicalPhenotypes(chest_regions=regions,
            chest_types=types, pairs=pairs)

    df = biomech_pheno.execute(lm, options.cid, spacing, J=J, ADI=ADI, SRI=SRI)

    if options.out_csv is not None:
        df.to_csv(options.out_csv, index=False)


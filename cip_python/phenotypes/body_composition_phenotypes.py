import numpy as np
from scipy.stats import mode, kurtosis, skew
from optparse import OptionParser
from cip_python.input_output import ImageReaderWriter
from cip_python.phenotypes import Phenotypes
from cip_python.common import ChestConventions
from cip_python.utils import RegionTypeParser

class BodyCompositionPhenotypes(Phenotypes):
    """General purpose class for generating body composition-based phenotypes.

    The user can specify chest regions, chest types, or region-type pairs over
    which to compute the phenotypes. Otherwise, the phenotypes will be computed
    over all structures in the specified labelmap. The following phenotypes are
    computed using the 'execute' method:
    'AxialCSA': Axial cross-sectional area
    'CoronalCSA': Coronal cross-sectional area
    'SagittalCSA': Sagittal cross-sectional area
    'HUMean': Mean value of the structure's HU values
    'HUStd': Standard deviation of the structure's HU values
    'HUKurtosis': Kurtosis of the structure's HU values. Fisher's definition is
    used, meaning that normal distribution has kurtosis of 0. The calculation
    is corrected for statistical bias.
    'HUSkewness': Skewness of the structure's HU values. The calculation is
    corrected for statistical bias.
    'HUMode': Mode of the structure's HU values
    'HUMedian': Median of the structure's HU values
    'HUMin': Min HU value for the structure
    'HUMax': Max HU value for the structure

    The following set of phenotypes are identical to the above except that
    computation is isolated to the HU interval [-50, 90]. These phenotypes are
    capture lean muscle information and only have meaning for muscle structures.
    'leanAxialCSA': Axial cross-sectional area 
    'leanCoronalCSA': Coronoal cross-sectional area
    'leanSagittalCSA': Sagitall cross-sectional area
    'leanHUMean': Mean value of the structure's HU values
    'leanHUStd': Standard deviation of the structure's HU values
    'leanHUKurtosis': Kurtosis of the structure's HU values. Fisher's definition is
    used, meaning that normal distribution has kurtosis of 0. The calculation
    is corrected for statistical bias.
    'leanHUSkewness': Skewness of the structure's HU values. The calculation is
    corrected for statistical bias.
    'leanHUMode': Mode of the structure's HU values
    'leanHUMedian': Median of the structure's HU values
    'leanHUMin': Min HU value for the structure
    'leanHUMax': Max HU value for the structure

    Parameters
    ----------
    chest_regions : array, shape ( R ), optional
        Array of integers, with each element in the interval [0, 255],
        indicating the chest regions over which to compute the phenotypes.

    chest_types : array, shape ( T ), optional
        Array of integers, with each element in the interval [0, 255],
        indicating the chest types over which to compute the phenotypes.

    pairs : array, shape ( P, 2 ), optional
        Array of chest-region chest-type pairs over which to compute the
        phenotypes. The first column indicates the chest region, and the
        second column indicates the chest type. Each element should be in the
        interal [0, 255].

    pheno_names : list of strings, optional
        Names of phenotypes to compute. These names must conform to the
        accepted phenotype names, listed above. If none are given, all
        will be computed.
    """
    def __init__(self, chest_regions=None, chest_types=None, pairs=None,
                 pheno_names=None):
        if chest_regions is not None:
            if len(np.shape(chest_regions)) != 1:
                raise ValueError(\
                'chest_regions must be a 1D array with elements in [0, 255]')
            if np.max(chest_regions) > 255 or np.min(chest_regions) < 0:
                raise ValueError(\
                'chest_regions must be a 1D array with elements in [0, 255]')
        if chest_types is not None:
            if len(np.shape(chest_types)) != 1:
                raise ValueError(\
                'chest_types must be a 1D array with elements in [0, 255]')
            if np.max(chest_types) > 255 or np.min(chest_types) < 0:
                raise ValueError(\
                'chest_types must be a 1D array with elements in [0, 255]')
        if pairs is not None:
            if len(np.shape(pairs)) != 2:
                raise ValueError(\
                'cpairs must be a 1D array with elements in [0, 255]')
            if np.max(pairs) > 255 or np.min(pairs) < 0:
                raise ValueError(\
                'chest_types must be a 1D array with elements in [0, 255]')
                
        self.chest_regions_ = chest_regions
        self.chest_types_ = chest_types
        self.pairs_ = pairs
        self.requested_pheno_names = pheno_names

        Phenotypes.__init__(self)    

    def declare_pheno_names(self):
        """Creates the names of the phenotypes to compute

        Returns
        -------
        names : list of strings
            Phenotype names
        """
        names = ['AxialCSA', 'CoronalCSA', 'SagittalCSA', 'HUMean', 'HUStd',
                 'HUKurtosis', 'HUSkewness', 'HUMode', 'HUMedian', 'HUMin',
                 'HUMax', 'leanAxialCSA', 'leanCoronalCSA', 'leanSagittalCSA',
                 'leanHUMean', 'leanHUStd', 'leanHUKurtosis', 'leanHUSkewness',
                 'leanHUMode', 'leanHUMedian', 'leanHUMin', 'leanHUMax']
        
        return names

    def get_cid(self):
        """Get the case ID (CID)

        Returns
        -------
        cid : string
            The case ID (CID)
        """
        return self.cid_

    def execute(self, ct, lm, cid, spacing, chest_regions=None,
                chest_types=None, pairs=None, pheno_names=None):
        """Compute the phenotypes for the specified structures for the
        specified threshold values.

        The following values are computed for the specified structures.
        'AxialCSA': Axial cross-sectional area
        'CoronalCSA': Coronoal cross-sectional area
        'SagittalCSA': Sagitall cross-sectional area        
        'HUMean': Mean value of the structure's HU values
        'HUStd': Standard deviation of the structure's HU values
        'HUKurtosis': Kurtosis of the structure's HU values. Fisher's definition
        is used, meaning that normal distribution has kurtosis of 0. The
        calculation is corrected for statistical bias.
        'HUSkewness': Skewness of the structure's HU values. The calculation is
        corrected for statistical bias.
        'HUMode': Mode of the structure's HU values
        'HUMedian': Median of the structure's HU values
        'HUMin': Min HU value for the structure
        'HUMax': Max HU value for the structure

        The following set of phenotypes are identical to the above except that
        computation is isolated to the HU interval [-50, 90]. These phenotypes
        are capture lean muscle information and only have meaning for muscle
        structures.
        'leanAxialCSA': Axial cross-sectional area 
        'leanCoronalCSA': Coronoal cross-sectional area
        'leanSagittalCSA': Sagitall cross-sectional area
        'leanHUMean': Mean value of the structure's HU values
        'leanHUStd': Standard deviation of the structure's HU values
        'leanHUKurtosis': Kurtosis of the structure's HU values. Fisher's
        definition is used, meaning that normal distribution has kurtosis of 0.
        The calculation is corrected for statistical bias.
        'leanHUSkewness': Skewness of the structure's HU values. The calculation
        is corrected for statistical bias.
        'leanHUMode': Mode of the structure's HU values
        'leanHUMedian': Median of the structure's HU values
        'leanHUMin': Min HU value for the structure
        'leanHUMax': Max HU value for the structure

        Parameters
        ----------
        ct : array, shape ( X, Y, Z )
            The 3D CT image array

        lm : array, shape ( X, Y, Z )
            The 3D label map array

        cid : string
            Case ID

        spacing : array, shape ( 3 )
            The x, y, and z spacing, respectively, of the CT volume
            
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
        assert len(ct.shape) == len(lm.shape), \
            "CT and label map are not the same dimension"    

        dim = len(ct.shape)
        for i in range(0, dim):
            assert ct.shape[0] == lm.shape[0], \
                "Disagreement in CT and label map dimension"

        assert type(cid) == str, "cid must be a string"
        self.cid_ = cid
        self._spacing = spacing

        phenos_to_compute = self.pheno_names_
        if pheno_names is not None:
            phenos_to_compute = pheno_names
        elif self.requested_pheno_names is not None:
            phenos_to_compute = self.requested_pheno_names

        rs = None
        ts = None
        ps = None
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
        if rs == None and ts == None and ps == None:
            rs = parser.get_all_chest_regions()
            ts = parser.get_chest_types()
            ps = parser.get_all_pairs()

        lean_ct_mask = np.logical_and(ct >= -50, ct <= 90)

        # Now compute the phenotypes and populate the data frame
        c = ChestConventions()
        if rs is not None:
            for r in rs:
                if r != 0:
                    mask = parser.get_mask(chest_region=r)
                    lean_mask = np.logical_and(mask, lean_ct_mask)
                    for n in phenos_to_compute:
                        if 'lean' in n:
                            self.add_pheno_group(ct, lean_mask,
                                c.GetChestRegionName(r),
                                c.GetChestWildCardName(), n)
                        else:
                            self.add_pheno_group(ct, mask,
                                c.GetChestRegionName(r),
                                c.GetChestWildCardName(), n)
        if ts is not None:
            for t in ts:
                if t != 0:
                    mask = parser.get_mask(chest_type=t)
                    lean_mask = np.logical_and(mask, lean_ct_mask)
                    for n in phenos_to_compute:
                        if 'lean' in n:
                            self.add_pheno_group(ct, lean_mask,
                                c.GetChestWildCardName(),
                                c.GetChestTypeName(t), n)
                        else:
                            self.add_pheno_group(ct, mask,
                                c.GetChestWildCardName(),
                                c.GetChestTypeName(t), n)
        if ps is not None:
            for p in ps:            
                if not (p[0] == 0 and p[1] == 0):
                    mask = parser.get_mask(chest_region=p[0], chest_type=p[1])
                    lean_mask = np.logical_and(mask, lean_ct_mask)
                    for n in phenos_to_compute:
                        if 'lean' in n:
                            self.add_pheno_group(ct, lean_mask,
                                c.GetChestRegionName(p[0]),
                                c.GetChestTypeName(p[1]), n)
                        else:
                            self.add_pheno_group(ct, mask,
                                c.GetChestRegionName(p[0]),
                                c.GetChestTypeName(p[1]), n)

        return self._df

    def add_pheno_group(self, ct, mask, chest_region, chest_type, pheno_name):
        """For a given mask, this function computes all phenotypes corresponding
        to the masked structure and adds them to the dataframe with the
        'add_pheno' method

        Parameters
        ----------
        ct : array, shape ( X, Y, Z )
            The 3D CT image array

        mask : boolean array, shape ( X, Y, Z )
            Boolean mask where True values indicate presence of the structure
            of interest

        chest_region : string
            Name of the chest region in the (region, type) key used to populate
            the dataframe

        chest_type : string
            Name of the chest region in the (region, type) key used to populate
            the dataframe

        pheno_name : string
            Name of the phenotype used to populate the dataframe
        """
        assert pheno_name in self.pheno_names_, "Invalid phenotype name"

        pheno_val = None
        mask_sum = np.sum(mask)
        if pheno_name == 'AxialCSA' or pheno_name == 'leanAxialCSA':
            pheno_val = self._spacing[0]*self._spacing[1]*mask_sum
        elif pheno_name == 'CoronalCSA' or pheno_name == 'leanCoronalCSA':
            pheno_val = self._spacing[0]*self._spacing[2]*mask_sum
        elif pheno_name == 'SagittalCSA' or pheno_name == 'leanSagittalCSA':
            pheno_val = self._spacing[1]*self._spacing[2]*mask_sum
        elif (pheno_name == 'HUMean' or pheno_name == 'leanHUMean') and \
            mask_sum > 0:
            pheno_val = np.mean(ct[mask])
        elif (pheno_name == 'HUStd' or pheno_name == 'leanHUStd') and \
            mask_sum > 0:
            pheno_val = np.std(ct[mask])
        elif (pheno_name == 'HUKurtosis' or pheno_name == 'leanHUKurtosis') \
            and mask_sum > 0:
            pheno_val = kurtosis(ct[mask], bias=False, fisher=True)
        elif (pheno_name == 'HUSkewness' or pheno_name == 'leanHUSkewness') \
            and mask_sum > 0:
            pheno_val = skew(ct[mask], bias=False)
        elif (pheno_name == 'HUMode' or pheno_name == 'leanHUMode') \
            and mask_sum > 0:
            min_val = np.min(ct[mask])
            pheno_val = np.argmax(np.bincount(ct[mask] + np.abs(min_val))) - \
                np.abs(min_val)
        elif (pheno_name == 'HUMedian' or pheno_name == 'leanHUMedian') and \
            mask_sum > 0:
            pheno_val = np.median(ct[mask])
        elif (pheno_name == 'HUMin' or pheno_name == 'leanHUMin') and \
            mask_sum > 0:
            pheno_val = np.min(ct[mask])
        elif (pheno_name == 'HUMax' or pheno_name == 'leanHUMax') and \
            mask_sum > 0:
            pheno_val = np.max(ct[mask])

        if pheno_val is not None:
            self.add_pheno([chest_region, chest_type], pheno_name, pheno_val)

if __name__ == "__main__":
    desc = """Generates body composition phenotypes given input CT and label \
    map data"""
    
    parser = OptionParser(description=desc)
    parser.add_option('--in_ct',
                      help='Input CT file', dest='in_ct', metavar='<string>',
                      default=None)
    parser.add_option('--in_lm',
                      help='Input label map containing structures of interest',
                      dest='in_lm', metavar='<string>', default=None)
    parser.add_option('--out_csv',
                      help='Output csv file in which to store the computed \
                      dataframe', dest='out_csv', metavar='<string>',
                      default=None)
    parser.add_option('--cid',
                      help='The case ID', dest='cid', metavar='<string>',
                      default=None)    
    parser.add_option('--pheno_names',
                      help='Comma separated list of phenotype value names to \
                      compute.', dest='pheno_names', metavar='<string>',
                      default=None)     

    (options, args) = parser.parse_args()

    image_io = ImageReaderWriter()

    lm, lm_header = image_io.read_in_numpy(options.in_lm)
    ct, ct_header = image_io.read_in_numpy(options.in_ct)

    # spacing = np.zeros(3)
    # spacing[0] = ct_header['space directions'][0][0]
    # spacing[1] = ct_header['space directions'][1][1]
    # spacing[2] = ct_header['space directions'][2][2]
    spacing = lm_header['spacing']

    pheno_names = None
    if options.pheno_names is not None:
        pheno_names = options.pheno_names.split(',')
    
    bc_pheno = BodyCompositionPhenotypes()    
    df = bc_pheno.execute(ct, lm, options.cid, spacing, pheno_names=pheno_names)
    
    if options.out_csv is not None:
        if pheno_names is None:
            df.to_csv(options.out_csv, index=False)
        else:
            cols = bc_pheno.static_names_handler_.keys()        
            for p in pheno_names:
                cols.append(p)
            for k in bc_pheno.key_names_:
                cols.append(k)        
            df[cols].to_csv(options.out_csv, index=False)

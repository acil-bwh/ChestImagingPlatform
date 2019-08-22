import numpy as np
from scipy.stats import mode, kurtosis, skew
from argparse import ArgumentParser
import warnings

from cip_python.phenotypes import Phenotypes
from cip_python.common import ChestConventions
from cip_python.input_output import ImageReaderWriter
from cip_python.utils import RegionTypeParser

class ParenchymaPhenotypes(Phenotypes):
    """General purpose class for generating parenchyma-based phenotypes.

    The user can specify chest regions, chest types, or region-type pairs over
    which to compute the phenotypes. Otherwise, the phenotypes will be computed
    over all structures in the specified labelmap. The following phenotypes are
    computed using the 'execute' method:
    'LAA950': fraction of the structure's region with CT HU values < -950
    'LAA925': fraction of the structure's region with CT HU values < -925
    'LAA910': fraction of the structure's region with CT HU values < -910
    'LAA905': fraction of the structure's region with CT HU values < -905
    'LAA900': fraction of the structure's region with CT HU values < -900
    'LAA875': fraction of the structure's region with CT HU values < -875
    'LAA856': fraction of the structure's region with CT HU values < -856
    'HAA700': fraction of the structure's region with CT HU values > -700
    'HAA600': fraction of the structure's region with CT HU values > -600
    'HAA500': fraction of the structure's region with CT HU values > -500
    'HAA250': fraction of the structure's region with CT HU values > -250
    'Perc10': HU value at the 10th percentile of the structure's HU histogram
    'Perc15': HU value at the 15th percentile of the structure's HU histogram
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
    'HUMean500': Mean CT value of the structure, but only considering CT
    values that are < -500 HU
    'HUStd500': Standard deviation of the structure's CT values, but only
    considering CT values that are < -500 HU
    'HUKurtosis500': Kurtosis of the structure's HU values, but only
    considering CT values that are < -500 HU
    'HUSkewness500': Skewness of the structure's HU values, but only
    considering CT values that are < -500 HU
    'HUMode500': Mode of the structure's HU values, but only
    considering CT values that are < -500 HU
    'HUMedian500': Median of the structure's HU values, but only
    considering CT values that are < -500 HU
    'HUMin500': Min HU value for the structure, but only considering CT values
    that are < -500 HU
    'HUMax500': Max HU value for the structure, but only considering CT values
    that are < -500 HU
    'HUMean950': Mean CT value of the structure, but only considering CT
    values that are < -950 HU
    'HUStd950': Standard deviation of the structure's CT values, but only
    considering CT values that are < -950 HU
    'HUKurtosis950': Kurtosis of the structure's HU values, but only
    considering CT values that are < -950 HU
    'HUSkewness950': Skewness of the structure's HU values, but only
    considering CT values that are < -950 HU
    'HUMode950': Mode of the structure's HU values, but only
    considering CT values that are < -950 HU
    'HUMedian950': Median of the structure's HU values, but only
    considering CT values that are < -950 HU
    'HUMin950': Min HU value for the structure, but only considering CT values
    that are < -950 HU
    'HUMax950': Max HU value for the structure, but only considering CT values
    that are < -950 HU
    'Volume': Volume of the structure, measured in liters
    'Mass': Mass of the structure measure in grams    
    'TypeFrac': The fraction of a type in a specified chest-region chest-type
    pair within the chest region of that pair.
    
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
        names = c.ParenchymaPhenotypeNames
        
        return names

    def get_cid(self):
        """Get the case ID (CID)

        Returns
        -------
        cid : string
            The case ID (CID)
        """
        return self.cid_

    def execute(self, lm, cid, spacing, ct=None, chest_regions=None,
                chest_types=None, pairs=None, pheno_names=None):
        """Compute the phenotypes for the specified structures for the
        specified threshold values.

        The following values are computed for the specified structures.
        'LAA950': fraction of the structure's region with CT HU values < -950
        'LAA925': fraction of the structure's region with CT HU values < -925
        'LAA910': fraction of the structure's region with CT HU values < -910
        'LAA905': fraction of the structure's region with CT HU values < -905
        'LAA900': fraction of the structure's region with CT HU values < -900
        'LAA875': fraction of the structure's region with CT HU values < -875
        'LAA856': fraction of the structure's region with CT HU values < -856
        'HAA700': fraction of the structure's region with CT HU values > -700
        'HAA600': fraction of the structure's region with CT HU values > -600
        'HAA500': fraction of the structure's region with CT HU values > -500
        'HAA250': fraction of the structure's region with CT HU values > -250
        'Perc10': HU value at the 10th percentile of the structure's HU
        histogram
        'Perc15': HU value at the 15th percentile of the structure's HU
        histogram
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
        'HUMean950': Mean CT value of the structure, but only considering CT
        values that are < -950 HU
        'HUStd950': Standard deviation of the structure's CT values, but only
        considering CT values that are < -950 HU
        'HUKurtosis950': Kurtosis of the structure's HU values, but only
        considering CT values that are < -950 HU
        'HUSkewness950': Skewness of the structure's HU values, but only
        considering CT values that are < -950 HU
        'HUMode950': Mode of the structure's HU values, but only
        considering CT values that are < -950 HU
        'HUMedian950': Median of the structure's HU values, but only
        considering CT values that are < -950 HU
        'HUMin950': Min HU value for the structure, but only considering CT values
        that are < -950 HU
        'HUMax950': Max HU value for the structure, but only considering CT values
        that are < -950 HU
        'HUMean500': Mean CT value of the structure, but only considering CT
        values that are < -500 HU
        'HUStd500': Standard deviation of the structure's CT values, but only
        considering CT values that are < -500 HU
        'HUKurtosis500': Kurtosis of the structure's HU values, but only
        considering CT values that are < -500 HU
        'HUSkewness500': Skewness of the structure's HU values, but only
        considering CT values that are < -500 HU
        'HUMode500': Mode of the structure's HU values, but only
        considering CT values that are < -500 HU
        'HUMedian500': Median of the structure's HU values, but only
        considering CT values that are < -500 HU
        'HUMin500': Min HU value for the structure, but only considering CT
        values that are < -500 HU
        'HUMax500': Max HU value for the structure, but only considering CT
        values that are < -500 HU
        'Volume': Volume of the structure, measured in liters
        'Mass': Mass of the structure measure in grams
        'TypeFrac': The fraction of a type in a specified chest-region 
        chest-type pair within the chest region of that pair.
        
        Parameters
        ----------
        lm : array, shape ( X, Y, Z )
            The 3D label map array

        cid : string
            Case ID

        spacing : array, shape ( 3 )
            The x, y, and z spacing, respectively, of the CT volume

        ct : array, shape ( X, Y, Z )
            The 3D CT image array
                        
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
        if ct is not None:
            assert len(ct.shape) == len(lm.shape), \
              "CT and label map are not the same dimension"    

        dim = len(lm.shape)
        if ct is not None:
            for i in range(0, dim):
                assert ct.shape[0] == lm.shape[0], \
                  "Disagreement in CT and label map dimension"

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

    def add_pheno_group(self, ct, mask, mask_region, mask_type, chest_region, 
                        chest_type, phenos_to_compute):
        """This function computes phenotypes and adds them to the dataframe with
        the 'add_pheno' method. 
        
        Parameters
        ----------
        ct : array, shape ( X, Y, Z )
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
        1. Schneider et al, 'Correlation between CT numbers and tissue
        parameters needed for Monte Carlo simulations of clinical dose
        distributions'
        """
        mask_sum = np.sum(mask)
        print ("Computing phenos for region %s and type %s"%(chest_region,chest_type))
        if ct is not None:
            ct_mask=ct[mask]
            hus=dict()

            ##########
            # Changed in Aug 21 2019 to speed phenotyping. To avoid continuous
            # evaluation labelmap
            ##########
            # PREVIUS CODE
            # hus[500] = ct[np.logical_and(mask, ct < -500)]
            # hus[950] = ct[np.logical_and(mask, ct < -950)]
            hus[500] = ct_mask[ct_mask < -500]
            hus[950] = ct_mask[ct_mask < -950]

            for pheno_name in phenos_to_compute:
                assert pheno_name in self.pheno_names_, \
                  "Invalid phenotype name " + pheno_name
                pheno_val = None
                if pheno_name == 'LAA950' and mask_sum > 0:
                    pheno_val = float(np.sum(ct_mask < -950.)) / mask_sum
                elif pheno_name == 'LAA925' and mask_sum > 0:
                    pheno_val = float(np.sum(ct_mask < -925.)) / mask_sum
                elif pheno_name == 'LAA910' and mask_sum > 0:
                    pheno_val = float(np.sum(ct_mask < -910.)) / mask_sum
                elif pheno_name == 'LAA905' and mask_sum > 0:
                    pheno_val = float(np.sum(ct_mask < -905.)) / mask_sum
                elif pheno_name == 'LAA900' and mask_sum > 0:
                    pheno_val = float(np.sum(ct_mask < -900.)) / mask_sum
                elif pheno_name == 'LAA875' and mask_sum > 0:
                    pheno_val = float(np.sum(ct_mask < -875.)) / mask_sum
                elif pheno_name == 'LAA856' and mask_sum > 0:
                    pheno_val = float(np.sum(ct_mask < -856.)) / mask_sum
                elif pheno_name == 'HAA700' and mask_sum > 0:
                    pheno_val = float(np.sum(ct_mask > -700.)) / mask_sum
                elif pheno_name == 'HAA600' and mask_sum > 0:
                    pheno_val = float(np.sum(ct_mask > -600)) / mask_sum
                elif pheno_name == 'HAA500' and mask_sum > 0:
                    pheno_val = float(np.sum(ct_mask > -500)) / mask_sum
                elif pheno_name == 'HAA250' and mask_sum > 0:
                    pheno_val = float(np.sum(ct_mask > -250)) / mask_sum
                elif pheno_name == 'Perc15' and mask_sum > 0:
                    pheno_val = np.percentile(ct_mask, 15)
                elif pheno_name == 'Perc10' and mask_sum > 0:
                    pheno_val = np.percentile(ct_mask, 10)
                elif pheno_name == 'HUMean' and mask_sum > 0:
                    pheno_val = np.mean(ct_mask)
                elif pheno_name == 'HUStd' and mask_sum > 0:
                    pheno_val = np.std(ct_mask)
                elif pheno_name == 'HUKurtosis' and mask_sum > 0:
                    pheno_val = kurtosis(ct_mask, bias=False, fisher=True)
                elif pheno_name == 'HUSkewness' and mask_sum > 0:
                    pheno_val = skew(ct_mask, bias=False)
                elif pheno_name == 'HUMode' and mask_sum > 0:
                    min_val = np.min(ct_mask.clip(-3000))
                    pheno_val = np.argmax(np.bincount(ct_mask.clip(-3000) + \
                        np.abs(min_val))) - np.abs(min_val)
                elif pheno_name == 'HUMedian' and mask_sum > 0:
                    pheno_val = np.median(ct_mask)
                elif pheno_name == 'HUMin' and mask_sum > 0:
                    pheno_val = np.min(ct_mask)
                elif pheno_name == 'HUMax' and mask_sum > 0:
                    pheno_val = np.max(ct_mask)
                elif pheno_name == 'HUMean500' and mask_sum > 0:
                    if hus[500].shape[0] > 0:
                        pheno_val = np.mean(hus[500])
                elif pheno_name == 'HUStd500' and mask_sum > 0:
                    if hus[500].shape[0] > 0:
                        pheno_val = np.std(hus[500])
                elif pheno_name == 'HUKurtosis500' and mask_sum > 0:
                    if hus[500].shape[0]:
                        pheno_val = kurtosis(hus[500], bias=False, fisher=True)
                elif pheno_name == 'HUSkewness500' and mask_sum > 0:
                    if hus[500].shape[0] > 0:
                        pheno_val = skew(hus[500], bias=False)
                elif pheno_name == 'HUMode500' and mask_sum > 0:
                    if hus[500].shape[0] > 0:
                        min_val = np.min(hus[500].clip(-3000))
                        pheno_val = np.argmax(np.bincount(hus[500].clip(-3000) +\
                            np.abs(min_val))) - np.abs(min_val)                
                elif pheno_name == 'HUMedian500' and mask_sum > 0:
                    if hus[500].shape[0] > 0:
                        pheno_val = np.median(hus[500])
                elif pheno_name == 'HUMin500' and mask_sum > 0:
                    if hus[500].shape[0] > 0:
                        pheno_val = np.min(hus[500])
                elif pheno_name == 'HUMax500' and mask_sum > 0:
                    if hus[500].shape[0] > 0:
                        pheno_val = np.max(hus[500])
                elif pheno_name == 'HUMean950' and mask_sum > 0:
                    if hus[950].shape[0] > 0:
                        pheno_val = np.mean(hus[950])
                elif pheno_name == 'HUStd950' and mask_sum > 0:
                    if hus[950].shape[0] > 0:
                        pheno_val = np.std(hus[950])
                elif pheno_name == 'HUKurtosis950' and mask_sum > 0:
                    if hus[950].shape[0]:
                        pheno_val = kurtosis(hus[950], bias=False, fisher=True)
                elif pheno_name == 'HUSkewness950' and mask_sum > 0:
                    if hus[950].shape[0] > 0:
                        pheno_val = skew(hus[950], bias=False)
                elif pheno_name == 'HUMode950' and mask_sum > 0:
                    if hus[950].shape[0] > 0:
                        min_val = np.min(hus[950].clip(-3000))
                        pheno_val = np.argmax(np.bincount(hus[950].clip(-3000) +\
                            np.abs(min_val))) - np.abs(min_val)
                elif pheno_name == 'HUMedian950' and mask_sum > 0:
                    if hus[950].shape[0] > 0:
                        pheno_val = np.median(hus[950])
                elif pheno_name == 'HUMin950' and mask_sum > 0:
                    if hus[950].shape[0] > 0:
                        pheno_val = np.min(hus[950])
                elif pheno_name == 'HUMax950' and mask_sum > 0:
                    if hus[950].shape[0] > 0:
                        pheno_val = np.max(hus[950])
                elif pheno_name == 'Volume':
                    #Value in liters
                    pheno_val = np.prod(self._spacing)*float(mask_sum)/1e6
                elif pheno_name == 'Mass' and mask_sum > 0:
                    # This quantity is computed in a piecewise linear form
                    # according to the prescription presented in ref. [1].
                    # Mass is computed in grams. First compute the
                    # contribution in HU interval from -98 and below.
                    pheno_val = 0.0
                    HU_tmp = ct[np.logical_and(mask, ct < -98)].clip(-1000)
                    if HU_tmp.shape[0] > 0:
                        m = (1.21e-3-0.93)/(-1000+98)
                        b = 1.21e-3 + 1000*m
                        pheno_val += np.sum((m*HU_tmp + b)*\
                            np.prod(self._spacing)*0.001)
        
                    # Now compute the mass contribution in the interval
                    # [-98, 18] HU. Note the in the original paper, the
                    # interval is defined from -98HU to 14HU, but we
                    # extend in slightly here so there are no gaps in
                    # coverage. The values we report in the interval
                    # [14, 23] should be viewed as approximate.
                    HU_tmp = \
                      ct[np.logical_and(np.logical_and(mask, ct >= -98),
                            ct <= 18)]
                    if HU_tmp.shape[0] > 0:
                        pheno_val += \
                          np.sum((1.018 + 0.893*HU_tmp/1000.0)*\
                            np.prod(self._spacing)*0.001)
        
                    # Compute the mass contribution in the interval
                    # (18, 100]
                    HU_tmp = \
                      ct[np.logical_and(np.logical_and(mask, ct > 18),
                                               ct <= 100)]
                    if HU_tmp.shape[0] > 0:
                        pheno_val += np.sum((1.003 + 1.169*HU_tmp/1000.0)*\
                            np.prod(self._spacing)*0.001)
        
                    # Compute the mass contribution in the interval > 100
                    HU_tmp = ct[np.logical_and(mask, ct > 100)]
                    if HU_tmp.shape[0] > 0:
                        pheno_val += np.sum((1.017 + 0.592*HU_tmp/1000.0)*\
                            np.prod(self._spacing)*0.001)
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

    parser = ArgumentParser(description=desc)
    
    parser.add_argument('--in_ct', '-in_ct', required=True,
                      help='Input CT file')

    parser.add_argument('--in_lm', '-in_lm', required=True,
                      help='Input label map containing structures of interest')

    parser.add_argument('--out_csv', '-out_csv', required=True,
                      help='Output csv file in which to store the computed dataframe')

    parser.add_argument('--cid', '-cid', required=True, help='Case id')

    # Keep these arguments as strings for compatibility purposes
    parser.add_argument('-r', dest='chest_regions',
                      help='Chest regions. Should be specified as a \
                      common-separated list of string values indicating the \
                      desired regions over which to compute the phenotypes. \
                      E.g. LeftLung,RightLung would be a valid input.')

    parser.add_argument('-t', dest='chest_types',
                      help='Chest types. Scd phhould be specified as a \
                      common-separated list of string values indicating the \
                      desired types over which to compute the phenotypes. \
                      E.g.: Vessel,NormalParenchyma would be a valid input.')
    parser.add_argument('-p', dest='pairs',
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
                      chest region.')

    args = parser.parse_args()

    image_io = ImageReaderWriter()

    regions = None
    if args.chest_regions is not None:
        regions = args.chest_regions.split(',')
    types = None
    if args.chest_types is not None:
        types = args.chest_types.split(',')

    lm, lm_header = image_io.read_in_numpy(args.in_lm)

    ct, ct_header = image_io.read_in_numpy(args.in_ct)

    spacing = lm_header['spacing']

    pairs = None
    if args.pairs is not None:
        tmp = args.pairs.split(',')
        assert len(tmp) % 2 == 0, 'Specified pairs not understood'
        pairs = []
        for i in range(0, len(tmp)/2):
            pairs.append([tmp[2*i], tmp[2*i+1]])

    paren_pheno = ParenchymaPhenotypes(chest_regions=regions, chest_types=types, pairs=pairs)

    df = paren_pheno.execute(lm, args.cid, spacing, ct=ct)

    df.to_csv(args.out_csv, index=False)

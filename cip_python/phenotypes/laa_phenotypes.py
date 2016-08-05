import numpy as np
from . import Phenotypes
from ..common import ChestConventions
from ..utils import RegionTypeParser

class LAAPhenotypes(Phenotypes):
    """Compute a low attenuating area (LAA) phenotype.

    This computes the percentage of the entity mask (chest region, chest type,
    or chest region-type pair) falling below specified threshold(s).

    Parameters
    ----------
    threshs : array, shape ( N ), optional
        Array of threshold values for computing low attenuating areas. If none
        specified, an array of standard values (-950, -910, -856) will be used.

    chest_regions : array, shape ( R ), optional
        Array of integers, with each element in the interval [0, 255],
        indicating the chest regions over which to compute the LAA

    chest_types : array, shape ( T ), optional
        Array of integers, with each element in the interval [0, 255],
        indicating the chest types over which to compute the LAA

    pairs : array, shape ( P, 2 ), optional
        Array of chest-region chest-type pairs over which to compute the LAA.
        The first column indicates the chest region, and the second column
        indicates the chest type. Each element should be in the interal
        [0, 255].
    """

    def __init__(self, threshs=None, chest_regions=None, chest_types=None,
                 pairs=None):
        
        self._chest_region_type_assert(chest_regions,chest_types,pairs)
        self.chest_regions_ = chest_regions
        self.chest_types_ = chest_types
        self.pairs_ = pairs

        self.threshs_ = None
        if threshs is None:
            self.threshs_ = np.array([-950, -910, -856], dtype=int)
        else:
            self.threshs_ = threshs

        Phenotypes.__init__(self)

    def declare_pheno_names(self):
        """Creates the names of the phenotypes to compute

        Returns
        -------
        names : list of strings
            Phenotype names
        """
        names = []
        for t in self.threshs_:
            names.append('LAA'+str(int(np.abs(np.round(t)))))

        return names

    def get_cid(self):
        """Get the case ID (CID)
        """
        return self.cid_

    def execute(self, ct, lm, cid, chest_regions=None, chest_types=None,
                pairs=None):
        """Compute the phenotypes for the specified structures for the
        specified threshold values.

        Parameters
        ----------
        ct : array, shape ( X, Y, Z )
            The 3D CT image array

        lm : array, shape ( X, Y, Z )
            The 3D label map array

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

        Returns
        -------
        df : pandas dataframe
            Dataframe containing info about machine, run time, and chest region
            chest type phenotype quantities.        
        """
        assert len(ct.shape) == len(lm.shape), \
            "CT and label map are not the same dimension"

        dim = len(ct.shape)
        for i in xrange(0, dim):
            assert ct.shape[0] == lm.shape[0], \
                "Disagreement in CT and label map dimension"

        assert type(cid) == str, "cid must be a string"
        self.cid_ = cid

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

        # Now compute the phenotypes and populate the data frame
        c = ChestConventions()
        if rs is not None:
            for r in rs:
                if r != 0:
                    mask = parser.get_mask(chest_region=r)
                    for tt in self.threshs_:
                        pheno_name = 'LAA' + str(int(np.abs(np.round(tt))))
                        pheno_val = float(np.sum(ct[mask] <= tt))/np.sum(mask)
                        self.add_pheno([c.GetChestRegionName(r),
                                        c.GetChestWildCardName()],
                                        pheno_name, pheno_val)                        
        if ts is not None:
            for t in ts:
                if t != 0:
                    mask = parser.get_mask(chest_type=t)
                    for tt in self.threshs_:
                        pheno_name = 'LAA' + str(int(np.abs(np.round(tt))))
                        pheno_val = float(np.sum(ct[mask] <= tt))/np.sum(mask)
                        self.add_pheno([c.GetChestWildCardName(),
                                        c.GetChestTypeName(t)],
                                        pheno_name, pheno_val)                            
        if ps is not None:
            for p in ps:            
                if not (p[0] == 0 and p[1] == 0):
                    mask = parser.get_mask(chest_region=p[0], chest_type=p[1])
                    for tt in self.threshs_:
                        pheno_name = 'LAA'+str(int(np.abs(np.round(tt))))
                        pheno_val = float(np.sum(ct[mask] <= tt))/np.sum(mask)
                        self.add_pheno([c.GetChestRegionName(p[0]),
                                        c.GetChestTypeName(p[1])],
                                        pheno_name, pheno_val)

        return self._df

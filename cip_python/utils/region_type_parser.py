import numpy as np
from cip_python.common import ChestConventions


class RegionTypeParser():
    """Parses the chest-region chest-type input data to identify all existing
    chest regions, chest types, and region-type pairs.

    Parameters
    ----------
    data : ND array
        The input data. Each value is assumed to be an unsigned short (16 bit)
        data type, where the least significant 8 bits encode the chest region,
        and the most significant 8 bits encode the chest type.

    Attributes
    ----------
    labels_ : array, shape ( M )
        The M unique labels in the data set
    """

    def __init__(self, data,prebuild_lookup_table=False):        
        self._data = data
        assert len(data.shape) > 0, "Empty data set"

        self.labels_ = np.unique(self._data)

        ##########
        # Changed in Aug 21 2019 to speed phenotyping. To avoid continuous
        # evaluation labelmap
        ##########
        ##Create lookup table of indices for each label
        self._data_indices = dict()
        if prebuild_lookup_table==True:
            for ll in set(self.labels_):
                self._data_indices[ll] = np.where(self._data==ll)

   
    def get_mask(self, chest_region=None, chest_type=None):
        """Get's boolean mask of all data indices that match the chest-region
        chest-type query.

        If only a chest region is specified, then all voxels in that region
        will be included in the mask, regardless of the voxel's chest type
        value (chest region hierarchy is honored). If only a type is specified,
        then all voxels having that type will be included in the mask,
        regardless of the voxel's chest region. If both a region and type are
        speficied, then only those voxels matching both the region and type
        will be included in the mask (the chest region hierarchy is honored).

        Parameters
        ----------
        chest_region : int
            Integer value in the interval [0, 255] that indicates the chest
            region

        chest_type : int
            Integer value in the interval [0, 255] that indicates the chest
            type

        Returns
        -------
        mask : array, shape ( X, Y, Z )
            Boolean mask of all data indices that match the chest-region
            chest-type query. The chest region hierarchy is honored.
        """

        if chest_region is not None:
            if type(chest_region) != int and type(chest_region) != np.int64 \
              and type(chest_region) != np.int32:
                raise ValueError(
                    'chest_region must be an int between 0 and 255 inclusive')
        if chest_type is not None:
            if type(chest_type) != int and type(chest_type) != np.int64 \
              and type(chest_type) != np.int32:
                raise ValueError(
                    'chest_type must be an int between 0 and 255 inclusive')        
        
        conventions = ChestConventions()
             
        mask_labels = []
        for l in self.labels_:
            r = conventions.GetChestRegionFromValue(l)
            t = conventions.GetChestTypeFromValue(l)

            if chest_region is not None and chest_type is not None:
                if t==chest_type and \
                  conventions.CheckSubordinateSuperiorChestRegionRelationship(\
                  r, chest_region):
                    mask_labels.append(l)
            elif t == chest_type:
                mask_labels.append(l)
            elif chest_region is not None:
                if conventions.\
                    CheckSubordinateSuperiorChestRegionRelationship(r, \
                    chest_region):
                    mask_labels.append(l)
                
        mask = np.empty(self._data.shape, dtype=bool)
        mask[:] = False


        ##########
        # Changed in Aug 21 2019 to speed phenotyping. To avoid continuous
        # evaluation labelmap
        ##########
        # PREVIUS CODE
        # for ll in set(mask_labels):
        #     tic = time.time()
        #     mask[self._data == ll] = True
        for ll in set(mask_labels):
            if ll in self._data_indices.keys():
                mask[self._data_indices[ll]] = True
            else:
                #Build cache of indices for label
                self._data_indices[ll] = np.where(self._data==ll)
                mask[self._data_indices[ll]] = True


        return mask

    def get_chest_regions(self):
        """Get the explicit list of chest regions in the data set.

        Returns
        -------
        chest_regions : array, shape ( N )
            Explicit list of the N chest regions in the data set
        """
        conventions = ChestConventions()

        tmp = []
        for l in self.labels_:            
            tmp.append(conventions.GetChestRegionFromValue(l))

        chest_regions = np.unique(np.array(tmp, dtype=int))
        return chest_regions

    def get_all_chest_regions(self):
        """Get all the chest regions in the data set, including those
        implicitly present as a result of the region hierarchy.

        Returns
        -------
        chest_regions : array, shape ( N )
            All chest regions in the data set
        """
        c = ChestConventions()
        num_regions = c.GetNumberOfEnumeratedChestRegions()

        tmp = []
        for l in self.labels_:
            r = c.GetChestRegionFromValue(l)

            for sup in range(0, num_regions):
                if c.CheckSubordinateSuperiorChestRegionRelationship(r, sup):
                    tmp.append(sup)

        chest_regions = np.unique(np.array(tmp, dtype=int))
        return chest_regions

    def get_chest_types(self):
        """Get the chest types in the data set
        
        Returns
        -------
        chest_types : array, shape ( N )
            All N chest types in the data set
        """
        c = ChestConventions()

        tmp = []
        for l in self.labels_:
            tmp.append(c.GetChestTypeFromValue(l))

        chest_types = np.unique(np.array(tmp, dtype=int))
        return chest_types

    def get_all_pairs(self):
        """Get all the region-type pairs, including implied pairs

        Returns
        -------
        pairs : array, shape ( N, 2 )
            All N chest-region chest-type pairs in the data set, including
            those implied by the region hierarchy. The first column indicates
            the chest region, and the second column represents the chest type.
        """
        c = ChestConventions()
        num_regions = c.GetNumberOfEnumeratedChestRegions()

        tmp = []
        for l in self.labels_:
            t = c.GetChestTypeFromValue(l)
            r = c.GetChestRegionFromValue(l)
            for sup in range(0, num_regions):
                if c.CheckSubordinateSuperiorChestRegionRelationship(r, sup):
                    if not (sup, t) in tmp:
                        tmp.append((sup, t))

        pairs = np.array(tmp, dtype=int)
        return pairs


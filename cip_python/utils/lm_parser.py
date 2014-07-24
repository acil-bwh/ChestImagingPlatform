import numpy as np
from cip_python.ChestConventions import ChestConventions

import pdb

class LMParser():
    """Parses the label information for a specified label map

    Parameters
    ----------
    lm : array, shape ( X, Y, Z )
        The 3D label map array

    Attributes
    ----------
    labels_ : array, shape ( N )
        The N unique labels in the label map
    """

    def __init__(self, lm):        
        self._lm = lm
        assert len(lm.shape) == 3, "Label map is not 3D"

        self.labels_ = np.unique(self._lm)

    def get_mask(self, chest_region=None, chest_type=None):
        """Get's boolean mask of all lm indices that match the chest-region
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
            Boolean mask of all lm indices that match the chest-region
            chest-type query. The chest region hierarchy is honored.
        """
        if chest_region is not None:
            if type(chest_region) != int:
                raise ValueError(
                    'chest_region must be an int between 0 and 255 inclusive')
        if chest_type is not None:
            if type(chest_type) != int:
                raise ValueError(
                    'chest_type must be an int between 0 and 255 inclusive')        
        
        conventions = ChestConventions()

        X = self._lm.shape[0]
        Y = self._lm.shape[1]
        Z = self._lm.shape[2]        
        
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

        mask = np.empty([X, Y, Z], dtype=bool)
        mask[:, :, :] = False

        for ml in mask_labels:
            mask = np.logical_or(mask, self._lm == ml)

        return mask

    def get_chest_regions(self):
        """Get the explicit list of chest regions in the label map.

        Returns
        -------
        chest_regions : array, shape ( N )
            Explicit list of the N chest regions in the label map
        """
        conventions = ChestConventions()

        tmp = []
        for l in self.labels_:            
            tmp.append(conventions.GetChestRegionFromValue(l))

        chest_regions = np.unique(np.array(tmp, dtype=int))
        return chest_regions

    def get_all_chest_regions(self):
        """Get all the chest regions in the label map, including those
        implicitly present as a result of the region hierarchy.

        Returns
        -------
        chest_regions : array, shape ( N )
            All chest regions in the label map
        """
        c = ChestConventions()
        num_regions = c.GetNumberOfEnumeratedChestRegions()

        tmp = []
        for l in self.labels_:
            r = c.GetChestRegionFromValue(l)

            for sup in xrange(0, num_regions):
                if c.CheckSubordinateSuperiorChestRegionRelationship(r, sup):
                    tmp.append(sup)

        chest_regions = np.unique(np.array(tmp, dtype=int))
        return chest_regions

    def get_chest_types(self):
        """Get the chest types in the label map

        Returns
        -------
        chest_types : array, shape ( N )
            All N chest types in the label map
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
            All N chest-region chest-type pairs in the label map, including
            those implied by the region hierarchy. The first column indicates
            the chest region, and the second column represents the chest type.
        """
        c = ChestConventions()
        num_regions = c.GetNumberOfEnumeratedChestRegions()

        tmp = []
        for l in self.labels_:
            t = c.GetChestTypeFromValue(l)
            r = c.GetChestRegionFromValue(l)
            for sup in xrange(0, num_regions):
                if c.CheckSubordinateSuperiorChestRegionRelationship(r, sup):
                    if not (sup, t) in tmp:
                        tmp.append((sup, t))

        pairs = np.array(tmp, dtype=int)
        return pairs

    def get_all_entities(self):
        """Get all the entities in the label map: all regions, all types, and
        all region-type pairs

        Returns
        -------
        entities : array, shape ( N, 2 )
            All N entities in the label map (regions, types, and region-type
            pairs). This includes regions implied by the region hierarchy.
        """
        types = self.get_chest_types()
        regions = self.get_all_chest_regions()
        pairs = self.get_all_pairs()

        tmp = []
        for t in types:
            if (0, t) not in tmp:
                tmp.append((0, t))
        for r in regions:
            if (r, 0) not in tmp:
                tmp.append((r, 0))
        for p in pairs:
            if (p[0], p[1]) not in tmp:
                tmp.append((p[0], p[1]))

        entities = np.array(tmp, dtype=int)
        return entities




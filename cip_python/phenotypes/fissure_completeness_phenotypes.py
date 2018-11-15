import numpy as np
from scipy.interpolate import Rbf
from argparse import ArgumentParser
import warnings

from cip_python.phenotypes import Phenotypes
from cip_python.common import ChestConventions
from cip_python.input_output import ImageReaderWriter

class FissureCompletenessPhenotypes(Phenotypes):
    """Class for computing lung fissure completeness.
    """
    def __init__(self):
        Phenotypes.__init__(self)    

    def declare_pheno_names(self):
        """Creates the names of the phenotypes to compute

        Returns
        -------
        names : list of strings
            Phenotype names
        """        
        c = ChestConventions()
        names = c.FissureCompletenessPhenotypeNames
        
        return names

    def get_cid(self):
        """Get the case ID (CID)

        Returns
        -------
        cid : string
            The case ID (CID)
        """
        return self.cid_

    def get_triangle_area(self, p1, p2, p3):
        """Computes the area of a triangle defined by three 3D points (p1, p2,
        p3) using Heron's formula.

        Parameters
        ----------
        p1 : array, shape ( 3 )
            Coordinates of point 1

        p2 : array, shape ( 3 )
            Coordinates of point 2

        p3 : array, shape ( 3 )
            Coordinates of point 3

        Returns
        -------
        area : float
            The area of the triangle            
        """
        # Compute the length of each side of the triangle
        l1 = np.sqrt(np.sum((p1 - p2)**2))
        l2 = np.sqrt(np.sum((p1 - p3)**2))
        l3 = np.sqrt(np.sum((p3 - p2)**2))        

        s = (l1 + l2 + l3)/2.

        area = np.sqrt(s*(s-l1)*(s-l2)*(s-l3))

        return area

    def get_fissure_completeness(surface, fissure):
        """Compute the completeness of the fissure. A thin-plate spline
        representation of the lobe boundary is constructed from the points in
        'surface'. For each point in the 'surface' list, the surface area of
        the TPS boundary is approximated and tallied. If the point is also in
        the 'fissure' coordinate list, the coordinate's surface area
        approximation is also added to the fissure surface area tally. The
        final completeness measure is the ratio of fissure_area/surface_area.

        Parameters
        ----------
        surface : list of 3D lists
            Each element is a 3D coordinate along a lobe boundary.
        
        fissure : list of 3D lists
            Each element is a 3D coordinate of a fissure voxel.

        Returns
        -------
        completeness : double
            In the interval [0, 1] with 0 being totally absent and 1 being
            totally complete.
        """
        surface_arr = np.array(surface)

        # Downsample the number of pooints in the surface list so as not to
        # choke the TPS computation.
        ids = (surface_arr[:, 0]%2 == 0) & (surface_arr[:, 1]%2 == 0)
        tps = Rbf(surface_arr[ids, 0], surface_arr[ids, 1],
            surface_arr[ids, 2], function='thin_plate')

        surface_area = 0.
        fissure_area = 0.

        # For each coordinate in i,j,k space, the corresponding TPS surface
        # area is approximate by computing the area of four triangles formed
        # by the index itself, and half-step offsets around the index.
        nw_delta = np.array([-0.5, -0.5, 0])
        ne_delta = np.array([0.5, -0.5, 0])
        sw_delta = np.array([-0.5, 0.5, 0])
        se_delta = np.array([0.5, 0.5, 0])
        patch_areas = []
        for i in xrange(0, surface_arr.shape[0]):
            m = surface_arr[i, :]
            nw = m + nw_delta
            nw[2] = tps(nw[0], nw[1])
        
            ne = m + ne_delta
            ne[2] = tps(ne[0], ne[1])
        
            sw = m + sw_delta
            sw[2] = tps(sw[0], sw[1])
        
            se = m + se_delta
            se[2] = tps(se[0], se[1])
        
            patch_area  = self.get_triangle_area(nw, m, sw) + \
                self.get_triangle_area(nw, m, ne) + \
                self.get_triangle_area(ne, m, se) + \
                self.get_triangle_area(m, sw, se)
            patch_areas.append(patch_area)
            surface_area += patch_area
                        
            if list(m) in fissure_list:
                fissure_area += patch_area

        completeness = fissure_area/surface_area                

        return completeness
        
    def execute(self, lm, cid):
        """Compute the fissure completeness phenotypes.
        
        Parameters
        ----------
        lm : array, shape ( X, Y, Z )
            The 3D lung lobe label map. It is assumed that when the lobe
            segmentation was produced, fissure particles were set in order to
            define those voxels that correspond to fissures.

        cid : string
            Case ID

        Returns
        -------
        df : pandas dataframe
            Dataframe containing info about machine, run time, and fissure
            completeness phenotype values
        """
        c = ChestConventions()

        assert type(cid) == str, "cid must be a string"
        self.cid_ = cid

        RUL = conventions.GetChestRegionValueFromName('RightSuperiorLobe')
        RML = conventions.GetChestRegionValueFromName('RightMiddleLobe')
        RLL = conventions.GetChestRegionValueFromName('RightInferiorLobe')
        LUL = conventions.GetChestRegionValueFromName('LeftSuperiorLobe')
        LLL = conventions.GetChestRegionValueFromName('LeftInferiorLobe')
        
        oblique = conventions.GetChestTypeValueFromName('ObliqueFissure')
        horizontal = conventions.GetChestTypeValueFromName('HorizontalFissure')
        
        nonzero_domain = np.where(np.sum(lm > 0, 2) > 0)
        ro_surface = []
        rh_surface = []
        lo_surface = []
        
        ro_fissure = []
        rh_fissure = []
        lo_fissure = []
        
        for i, j in zip(nonzero_domain[0], nonzero_domain[1]):
            last_region = 0
            ks = np.where(lm[i, j, :] > 0)[0]
        
            for k in xrange(0, ks.shape[0]):
                if k > 0:
                    if ks[k] - ks[k-1] > 1:
                        last_region = 0
                    
                curr_region = conventions.GetChestRegionFromValue(lm[i, j, k])
                curr_type = conventions.GetChestTypeFromValue(lm[i, j, k])
                if curr_type == oblique and (curr_region == RUL or \
                    curr_region == RML or curr_region == RLL):
                    ro_fissure.append([i, j, k]) 
                if curr_type == horizontal and (curr_region == RUL or \
                    curr_region == RML or curr_region == RLL):
                    rh_fissure.append([i, j, k])
                if curr_type == oblique and (curr_region == LUL or \
                    curr_region == LLL):
                    lo_fissure.append([i, j, k])
                    
                if curr_region != last_region:
                    if (last_region == RLL and curr_region == RUL) or \
                      (last_region == RLL and curr_region == RML) or \
                      (last_region == RUL and curr_region == RLL) or \
                      (last_region == RML and curr_region == RLL):
                        ro_surface.append([i, j, k])
                        break
                        
                    if (last_region == RML and curr_region == RUL) or \
                      (last_region == RUL and curr_region == RML):
                        rh_surface.append([i, j, k])
                        break
                        
                    if (last_region == LUL and curr_region == LLL) or \
                      (last_region == LLL and curr_region == LUL):
                        lo_surface.append([i, j, k])
                        break
                    
                last_region = curr_region

        lo_completeness = np.nan
        ro_completeness = np.nan
        rh_completeness = np.nan                
        if len(lo_surface) > 2:
            lo_completeness = \
            self.get_fissure_completeness(lo_surface, lo_fissure)
            
        if len(ro_surface) > 2:
            ro_completeness = \
            self.get_fissure_completeness(ro_surface, ro_fissure)

        if len(rh_surface) > 2:
            rh_completeness = \
            self.get_fissure_completeness(rh_surface, rh_fissure)            
                
        self.add_pheno(['LEFTLUNG', 'OBLIQUEFISSURE'],
            'LeftObliqueFissureCompleteness', lo_completeness)
        self.add_pheno(['RIGHTLUNG', 'OBLIQUEFISSURE'],
            'RightObliqueFissureCompleteness', ro_completeness)
        self.add_pheno(['RIGHTLUNG', 'HORIZONTALFISSURE'],
            'RightHorizontalFissureCompleteness', rh_completeness)    

        return self._df

if __name__ == "__main__":
    desc = """Computes lung fissure completeness measures."""

    parser = ArgumentParser(description=desc)
    
    parser.add_argument('--in_lm', '-in_lm', required=True,
        help='Lung lobe segmentation mask. It is assumed that when the lobe \
        segmentation was produced, fissure particles were set in order to \
        define those voxels that correspond to fissures')
    parser.add_argument('--out_csv', '-out_csv', required=True,
        help='Output csv file in which to store the computed dataframe')
    parser.add_argument('--cid', '-cid', required=True, help='Case id')

    args = parser.parse_args()

    image_io = ImageReaderWriter()
    lm, lm_header = image_io.read_in_numpy(args.in_lm)

    completeness_phenos = FissureCompletenessPhenotypes()
    df = completeness_phenos.execute(lm, args.cid)
    df.to_csv(args.out_csv, index=False)
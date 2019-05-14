import numpy as np
from scipy.interpolate import Rbf
from argparse import ArgumentParser
import warnings
import pandas as pd
import vtk, pdb
from cip_python.phenotypes import Phenotypes
from cip_python.common import ChestConventions
from cip_python.input_output import ImageReaderWriter

class FissurePhenotypes(Phenotypes):
    """Class for computing lung fissure completeness.
    """
    def __init__(self):
        Phenotypes.__init__(self)    
        self._lo_obb_tree = None
        self._ro_obb_tree = None
        self._rh_obb_tree = None
        
        self.df_data_detail_ = None
        
        self._conventions = ChestConventions()

        self._right_lung_value = self._conventions.\
          GetChestRegionValueFromName('RightLung')
        self._right_upper_lobe_value = self._conventions.\
          GetChestRegionValueFromName('RightSuperiorLobe')
        self._right_middle_lobe_value = self._conventions.\
          GetChestRegionValueFromName('RightMiddleLobe')
        self._right_lower_lobe_value = self._conventions.\
          GetChestRegionValueFromName('RightInferiorLobe')
        self._left_lung_value = self._conventions.\
          GetChestRegionValueFromName('LeftLung')        
        self._left_upper_lobe_value = self._conventions.\
          GetChestRegionValueFromName('LeftSuperiorLobe')
        self._left_lower_lobe_value = self._conventions.\
          GetChestRegionValueFromName('LeftInferiorLobe')
        self._oblique_fissure_value = self._conventions.\
          GetChestTypeValueFromName('ObliqueFissure')
        self._horizontal_fissure_value = self._conventions.\
          GetChestTypeValueFromName('HorizontalFissure')
        
    def declare_pheno_names(self):
        """Creates the names of the phenotypes to compute

        Returns
        -------
        names : list of strings
            Phenotype names
        """
        conventions = ChestConventions()
        names = conventions.FissurePhenotypeNames
        
        return names

    def patch_periphery_poly(self, df, irad, spacing):
        """This function marches through all those points that were deemed
        incomplete and evaluates whether or not they are close enough to the
        fissure surface to be considered fissures as well. This is needed
        because the fissure surface constructed from particles that is used
        to evaluate whether or not a point is a fissure does not
        completely represent the true fissure surface at the periphery. This
        is because those particles at the periphery "cover" more area than is
        represented by the triangular mesh surface.
        
        Paramters
        ---------
        df : pandas dataframe
            Contains fissure point detailed information
        
        irad : float
            Indicates spacing (on average) between particles

        spacing : array, shape ( 3 )
            Physical voxel spacing
        """
        not_fissure_indices = np.where(~df.is_fissure.values)[0]
        fissure_ids = df.is_fissure.values
        fissure_pts_mat = df[fissure_ids][['i', 'j', 'k']].values*spacing
        for nn in not_fissure_indices:
            pt = df[['i', 'j', 'k']].values[nn]*spacing
            if np.min(np.sqrt(np.sum((fissure_pts_mat - pt)**2, 1))) <= irad:
                df.at[nn, 'is_fissure'] = True
                                
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

    def init_fissure_surface(self, poly, fissure_type, alpha=None):
        """
        
        Parameters
        ----------
        poly : vtkPolyData

        fissure_type : string
            Either 'left_oblique', 'right_oblique', or 'right_horizontal'

        alpha : float, optional
            Will be retrieved from the input polydata (irad FieldDataArray)
            unless speficied. If the irad array does not exist, and if no
            value is passed, a default value of 1.5 will be used. This value
            relates to how closely packed the fissure points are that define
            the surface. The larger the value used, the farther apart two
            points can be and still be considered to represent a continuous,
            unbroken stretch of fissure.
        """
        if fissure_type not in ['left_oblique', 'right_oblique',
                                'right_horizontal']:
            raise ValueError('Unrecognized fissure type')

        if alpha is None:
            irad_arr = poly.GetFieldData().GetAbstractArray('irad')
            if irad_arr is not None:
                alpha = irad_arr.GetTuple(0)[0]
            else:
                alpha = 1.5
                
        delaunay = vtk.vtkDelaunay2D()
        delaunay.SetInputData(poly);
        delaunay.SetAlpha(alpha)
        delaunay.Update()

        surf_filter = vtk.vtkDataSetSurfaceFilter()
        surf_filter.SetInputData(delaunay.GetOutput())
        surf_filter.Update()
        
        if fissure_type == 'left_oblique':
            self._lo_obb_tree = vtk.vtkOBBTree()
            self._lo_obb_tree.SetDataSet(surf_filter.GetOutput())            
            self._lo_obb_tree.BuildLocator()
        elif fissure_type == 'right_oblique':
            self._ro_obb_tree = vtk.vtkOBBTree()
            self._ro_obb_tree.SetDataSet(surf_filter.GetOutput())
            self._ro_obb_tree.BuildLocator()
        else:
            self._rh_obb_tree = vtk.vtkOBBTree()
            self._rh_obb_tree.SetDataSet(surf_filter.GetOutput())
            self._rh_obb_tree.BuildLocator()            
    
    def is_fissure(self, index, lm, fissure_type, origin, spacing, ray_tol):
        """Determines whether or not the label map index coordinate corresponds
        to fissure or not.
        
        Parameters
        ----------
        index : 3D tuple
            The x, y, and z index coordinates to test.
        
        lm : array, shape ( X, Y, Z )
            The 3D lung lobe label map. 

        fissure_type : string
            Must be either 'left_oblique', 'right_oblique', or
            'right_horizontal'
        
        origin : array, shape (3)
            The origin (in physical coordinates) of the label map image

        spacing : array, shape (3)
            Voxel spacing in x, y, and z direction

        ray_tol : float
            Distance tolerance. The physical coordinate corresponding to the
            passed index must be within this vertical distance of the surface
            defined by the class's fissure points poly data (if any). If
            fissure point polydata are not being used to define the fissure
            surface, this parameter is irrelevant.            
        """
        if fissure_type not in ['left_oblique', 'right_oblique',
                                'right_horizontal']:
            raise ValueError('Unrecognized fissure type')

        cip_region = self._conventions.\
            GetChestRegionFromValue(int(lm[index[0], index[1], index[2]]))
        cip_type = self._conventions.\
            GetChestTypeFromValue(int(lm[index[0], index[1], index[2]]))

        ray_from = np.zeros(3)
        ray_from[0] = index[0]*spacing[0] + origin[0]
        ray_from[1] = index[1]*spacing[1] + origin[1]
        ray_from[2] = index[2]*spacing[2] + origin[2] - ray_tol

        ray_to = np.zeros(3)
        ray_to[0] = index[0]*spacing[0] + origin[0]
        ray_to[1] = index[1]*spacing[1] + origin[1]
        ray_to[2] = index[2]*spacing[2] + origin[2] + ray_tol

        intersection_points = vtk.vtkPoints()
        if fissure_type == 'right_oblique':
            if self._ro_obb_tree is not None:
                code = self._ro_obb_tree.IntersectWithLine(ray_from, ray_to,
                    intersection_points, None)
                return code != 0
            elif cip_type == self._oblique_fissure_value and \
            (cip_region == self._right_upper_lobe_value or \
            cip_region == self._right_middle_lobe_value or \
            cip_region == self._right_lower_lobe_value or \
            cip_region == self._right_lung):
                return True
        elif fissure_type == 'right_horizontal':
            if self._rh_obb_tree is not None:
                code = self._rh_obb_tree.IntersectWithLine(ray_from, ray_to,
                    intersection_points, None)
                return code != 0
            elif cip_type == self._horizontal_fissure_value and \
              (cip_region == self._right_upper_lobe_value or \
               cip_region == self._right_middle_lobe_value or \
               cip_region == self._right_lower_lobe_value or \
               cip_region == self._right_lung):
                return True             
        else:
            if self._lo_obb_tree is not None:
                code = self._lo_obb_tree.IntersectWithLine(ray_from, ray_to,
                        intersection_points, None)
                return code != 0
            elif cip_type == self._oblique_fissure_value and \
              (cip_region == self._left_upper_lobe_value or \
               cip_region == self._left_lower_lobe_value or \
               cip_region == self._left_lung):
                return True  

        return False

    def get_fissure_completeness_detail(self, surface, fissure, spacing,
        origin, completeness_type, cip_region, cip_type, tps_downsample=1):
        """Creates and returns a pandas data frame that records detailed info
        for each lobe boundary voxel considered, including the voxel
        coordinates, patch area, whether or not the boundary voxel is
        identified as a fissure, and CIP region and type. A thin-plate spline
        representation of the lobe boundary is constructed from the points in
        'surface'. For each point in the 'surface' list, the surface area of
        the TPS boundary is approximated and tallied. If the point is also in
        the 'fissure' coordinate list, the voxel is recorded as being
        identified as a fissure.

        Parameters
        ----------
        surface : list of 3D lists
            Each element is a 3D coordinate along a lobe boundary.
        
        fissure : list of 3D lists
            Each element is a 3D coordinate of a fissure voxel.

        spacing : array, shape (3)
            Voxel spacing in x, y, and z direction

        origin : array, shape (3)

            
        completeness_type : string, optional
            Either 'surface', in which case the surface area of the lobe
            boundaries and fissure regions will be compared, or 'count', in
            which case only the voxel counts of the boundary and fissure 
            regions will be compared.

        cip_region : string
            CIP region. Either 'LeftLung' or 'RightLung'
        
        cip_type : string
            CIP type. Either 'ObliqueFissure' or 'HorizontalFissure'
            
        tps_downsample : int, optional
            The amount by which to downsample the surface points before
            computing the TPS (1 -> no downsampling, 2 -> half the points will
            be used, etc). This option is irrelevant if 'completeness_type' is
            set to 'count'.
           
        Returns
        -------
        df : Pandas dataframe
            For each boundary voxel considered, this data frame records the
            (i, j, k) coordinates, whether or not the voxel was identified as
            a fissure, and the area patch area (either the approximate surface
            area of the boundary, or the axial-plane voxel area, depending on
            the value of 'completeness_type'). Also records the CIP region and
            type.
        """
        num_patches = len(surface)
        surface_arr = np.array(surface)
        is_fissure_arr = np.zeros(num_patches, dtype='bool')
        area_arr = np.nan*np.ones(num_patches)
        
        df_data_detail = \
          pd.DataFrame({'i': surface_arr[:, 0],
                        'j': surface_arr[:, 1],
                        'k': surface_arr[:, 2],
                        'Region': np.array([cip_region]*num_patches),
                        'Type': np.array([cip_type]*num_patches)})    

        surface_dom = [[s[0], s[1]] for s in surface]
        fissure_dom = [[f[0], f[1]] for f in fissure]
        
        if completeness_type == 'count':
            for i in range(0, surface_arr.shape[0]):
                area_arr[i] = spacing[0]*spacing[1]
                if surface_dom[i] in fissure_dom:
                    is_fissure_arr[i] = True
        else:        
            # Downsample the number of points in the surface list so as not to
            # choke the TPS computation.
            ids = np.arange(surface_arr.shape[0])%tps_downsample == 0
            tps = Rbf(surface_arr[ids, 0]*spacing[0],
                      surface_arr[ids, 1]*spacing[1],
                      surface_arr[ids, 2]*spacing[2], function='thin_plate')

            # For each coordinate in i,j,k space, the corresponding TPS surface
            # area is approximate by computing the area of four triangles
            # formed by the index itself, and half-step offsets around the
            # index.
            nw_delta = np.array([-spacing[0]/2., -spacing[1]/2., 0])
            ne_delta = np.array([spacing[0]/2., -spacing[1]/2., 0])
            sw_delta = np.array([-spacing[0]/2., spacing[1]/2., 0])
            se_delta = np.array([spacing[0]/2., spacing[1]/2., 0])
            for i in range(0, surface_arr.shape[0]):
                m = surface_arr[i, :]*spacing
                nw = m + nw_delta
                nw[2] = tps(nw[0], nw[1])
        
                ne = m + ne_delta
                ne[2] = tps(ne[0], ne[1])
        
                sw = m + sw_delta
                sw[2] = tps(sw[0], sw[1])
        
                se = m + se_delta
                se[2] = tps(se[0], se[1])

                area_arr[i] = self.get_triangle_area(nw, m, sw) + \
                    self.get_triangle_area(nw, m, ne) + \
                    self.get_triangle_area(ne, m, se) + \
                    self.get_triangle_area(m, sw, se)
            
                if surface_dom[i] in fissure_dom:
                    is_fissure_arr[i] = True
                
        df_data_detail = df_data_detail.assign(is_fissure=\
            pd.Series(is_fissure_arr, index=df_data_detail.index))
        df_data_detail = df_data_detail.assign(area=\
            pd.Series(area_arr, index=df_data_detail.index))
                
        return df_data_detail
    
    def execute(self, lm, origin, spacing, cid, completeness_type='surface',
                tps_downsample=1, lop_poly=None, rop_poly=None, rhp_poly=None,
                alpha=None, ray_tol=None):
        """Compute the fissure completeness phenotypes.
        
        Parameters
        ----------
        lm : array, shape ( X, Y, Z )
            The 3D lung lobe label map. 

        origin : array, shape (3)
            The origin (in physical coordinates) of the label map image
            
        spacing : array, shape (3)
            Voxel spacing in x, y, and z direction
        
        cid : string
            Case ID

        completeness_type : string, optional
            Either 'surface', in which case the surface area of the lobe
            boundaries and fissure regions will be compared, or 'count', in
            which case only the voxel counts of the boundary and fissure 
            regions will be compared.

        tps_downsample : int, optional
            The amount by which to downsample the surface points before
            computing the TPS (1 -> no downsampling, 2 -> half the points will
            be used, etc). This option is irrelevant if 'completeness_type' is
            set to 'count'.

        lop_poly : vtkPolyData, optional
            vtkPolyData containgin points defining the left oblique fissure
            surface. If specified, the fissure surface defined by these points
            will override the surface defined in the label map image.

        rop_poly : vtkPolyData, optional
            vtkPolyData containgin points defining the right oblique fissure
            surface. If specified, the fissure surface defined by these points
            will override the surface defined in the label map image.
        
        rhp_poly : vtkPolyData, optional
            vtkPolyData containgin points defining the right horiztonal fissure
            surface. If specified, the fissure surface defined by these points
            will override the surface defined in the label map image.

        alpha : double, optional
            alpha parameter for vtkDelaunay filter. Only relevant if
            vtkPolyData are passed for fissure definition.
            
        Returns
        -------
        df : pandas dataframe
            Dataframe containing info about machine, run time, and fissure
            completeness phenotype values
        """
        assert type(cid) == str, "cid must be a string"
        self.cid_ = cid

        if ray_tol is None:
            ray_tol = 1.0
        
        # Set up fissure surfaces if specified with polydata.
        if lop_poly is not None:
            self.init_fissure_surface(lop_poly, 'left_oblique', alpha)            
        if rop_poly is not None:
            self.init_fissure_surface(rop_poly, 'right_oblique', alpha)
        if rhp_poly is not None:
            self.init_fissure_surface(rhp_poly, 'right_horizontal', alpha)

        nonzero_count = np.where(np.sum(lm > 0, 2) > 0)
        ro_surface = []
        rh_surface = []
        lo_surface = []
        
        ro_fissure = []
        rh_fissure = []
        lo_fissure = []
        
        for i, j in zip(nonzero_count[0], nonzero_count[1]):
            last_region = 0
            for k in range(0, lm.shape[2]):                                    
                curr_region = self._conventions.\
                  GetChestRegionFromValue(int(lm[i, j, k]))
                curr_type = self._conventions.\
                  GetChestTypeFromValue(int(lm[i, j, k]))
                                      
                if curr_region != last_region:
                    if (last_region == self._right_lower_lobe_value and \
                        curr_region == self._right_upper_lobe_value) or \
                      (last_region == self._right_lower_lobe_value and \
                       curr_region == self._right_middle_lobe_value) or \
                      (last_region == self._right_lower_lobe_value and \
                       curr_region == self._right_lung_value): 
                        ro_surface.append([i, j, k])                        
                        if self.is_fissure([i, j, k], lm, 'right_oblique',
                          origin, spacing, ray_tol) or \
                          self.is_fissure([i, j, k-1], lm, 'right_oblique',
                          origin, spacing, ray_tol):
                            ro_fissure.append([i, j, k])
                        
                    if (last_region == self._right_middle_lobe_value and \
                        curr_region == self._right_upper_lobe_value) or \
                      (last_region == self._right_middle_lobe_value and \
                       curr_region == self._right_lung_value):
                        rh_surface.append([i, j, k])
                        if self.is_fissure([i, j, k], lm, 'right_horizontal',
                          origin, spacing, ray_tol) or \
                          self.is_fissure([i, j, k-1], lm, 'right_horizontal',
                          origin, spacing, ray_tol):
                            rh_fissure.append([i, j, k])
                        
                    if (last_region == self._left_lower_lobe_value and \
                        curr_region == self._left_upper_lobe_value) or \
                      (last_region == self._left_lower_lobe_value and \
                       curr_region == self._left_lung_value):
                        lo_surface.append([i, j, k])
                        if self.is_fissure([i, j, k], lm, 'left_oblique',
                          origin, spacing, ray_tol) or \
                          self.is_fissure([i, j, k-1], lm, 'left_oblique',
                          origin, spacing, ray_tol):
                            lo_fissure.append([i, j, k])                        
                    
                last_region = curr_region
        
        lo_completeness = np.nan
        ro_completeness = np.nan
        rh_completeness = np.nan
        df_lo_detail = None
        df_ro_detail = None
        df_rh_detail = None                
        if len(lo_surface) > 2:
            df_lo_detail = \
                self.get_fissure_completeness_detail(lo_surface, lo_fissure,
                    spacing, origin, completeness_type, 'LeftLung',
                    'ObliqueFissure', tps_downsample)
            if lop_poly is not None:
                irad = lop_poly.GetFieldData().GetAbstractArray('irad').\
                  GetTuple(0)[0]
                self.patch_periphery_poly(df_lo_detail, irad, spacing)
            
            ids = df_lo_detail.is_fissure.values == True
            lo_completeness = np.sum(df_lo_detail.area.values[ids])/\
              np.sum(df_lo_detail.area.values)
        if len(ro_surface) > 2:
            df_ro_detail = \
                self.get_fissure_completeness_detail(ro_surface, ro_fissure,
                    spacing, origin, completeness_type, 'RightLung',
                    'ObliqueFissure', tps_downsample)
            if rop_poly is not None:
                irad = rop_poly.GetFieldData().GetAbstractArray('irad').\
                  GetTuple(0)[0]
                self.patch_periphery_poly(df_ro_detail, irad, spacing)
            
            ids = df_ro_detail.is_fissure.values == True
            ro_completeness = np.sum(df_ro_detail.area.values[ids])/\
              np.sum(df_ro_detail.area.values)
        if len(rh_surface) > 2:
            df_rh_detail = \
                self.get_fissure_completeness_detail(rh_surface, rh_fissure,
                    spacing, origin, completeness_type, 'RightLung',
                    'HorizontalFissure', tps_downsample)
            if rhp_poly is not None:
                irad = rhp_poly.GetFieldData().GetAbstractArray('irad').\
                  GetTuple(0)[0]
                self.patch_periphery_poly(df_rh_detail, irad, spacing)            
            
            ids = df_rh_detail.is_fissure.values == True
            rh_completeness = np.sum(df_rh_detail.area.values[ids])/\
              np.sum(df_rh_detail.area.values)
              
        self.add_pheno(['LeftLung', 'ObliqueFissure'],
                       'Completeness', lo_completeness)
        self.add_pheno(['RightLung', 'ObliqueFissure'],
                        'Completeness', ro_completeness)
        self.add_pheno(['RightLung', 'HorizontalFissure'],
                        'Completeness', rh_completeness)    
                            
        self.df_data_detail_ = \
          pd.concat([df_lo_detail, df_ro_detail, df_rh_detail],
                    ignore_index=True) 

        return self._df

if __name__ == "__main__":
    desc = """Computes lung fissure completeness measures."""

    parser = ArgumentParser(description=desc)
    
    parser.add_argument('--in_lm', dest='in_lm', required=True,
        help='Lung lobe segmentation mask.')
    parser.add_argument('--out_csv', dest='out_csv', required=True,
        help='Output csv file in which to store the computed dataframe')
    parser.add_argument('--cid', dest='cid', required=True, help='Case id')
    parser.add_argument('--down', dest='down', required=False, default=1.,
        help='The amount by which to downsample the surface points before \
        computing the TPS (1 -> no downsampling, 2 -> half the points will \
        be used, etc). This option is irrelevant if completeness type is \
        set to count.')
    parser.add_argument('--type', dest='type', required=False,
        default='surface', help='Either surface, in which case the surface \
        area of the lobe boundaries and fissure regions will be compared, or \
        count, in which case only the voxel counts of the boundary and \
        fissure regions will be compared.')
    parser.add_argument('--lop', dest='lop', required=False, default=None,
        help='Left oblique points file (vtk). If specified, the surface \
        defined by these points will override the surface defined in the \
        label map image.')
    parser.add_argument('--rop', dest='rop', required=False, default=None,
        help='Right oblique points file (vtk). If specified, the surface \
        defined by these points will override the surface defined in the \
        label map image.')
    parser.add_argument('--rhp', dest='rhp', required=False, default=None,
        help='Right horizontal points file (vtk). If specified, the surface \
        defined by these points will override the surface defined in the \
        label map image.')
    parser.add_argument('--alpha', dest='alpha', required=False, default=None,
        help='alpha parameter for vtkDelaunay filter. Only relevant if \
        vtkPolyData are passed for fissure definition.')
    parser.add_argument('--detail', dest='detail', required=False, \
        default=None, help='csv output file name of voxel-by-voxel \
        information for all lobe boundary points considered.')
    parser.add_argument('--ray_tol', dest='ray_tol', required=False, \
        default=None, help='A particle mesh point must be within this \
        distance of the lobe boundary to be considered fissure. If fissure \
        point polydata are not being used to define the fissure surface, \
        this parameter is irrelevant.')
        
    op = parser.parse_args()

    image_io = ImageReaderWriter()
    lm, lm_header = image_io.read_in_numpy(op.in_lm)

    if 'spacing' in lm_header.keys():
        spacing = lm_header['spacing']
    elif 'space directions' in lm_header.keys():
        spacing = np.diag(lm_header['space directions'])
        
    lop_poly = None
    if op.lop is not None:
        lop_reader = vtk.vtkPolyDataReader()
        lop_reader.SetFileName(op.lop)
        lop_reader.Update()
        lop_poly = lop_reader.GetOutput()

    rop_poly = None
    if op.rop is not None:
        rop_reader = vtk.vtkPolyDataReader()
        rop_reader.SetFileName(op.rop)
        rop_reader.Update()
        rop_poly = rop_reader.GetOutput()

    rhp_poly = None
    if op.rhp is not None:
        rhp_reader = vtk.vtkPolyDataReader()
        rhp_reader.SetFileName(op.rhp)
        rhp_reader.Update()
        rhp_poly = rhp_reader.GetOutput()        

    alpha = None
    if op.alpha is not None:
        alpha = float(op.alpha)
        
    completeness_phenos = FissurePhenotypes()
    df = completeness_phenos.execute(lm, lm_header['space origin'],
        spacing, op.cid, op.type, int(op.down),
        lop_poly=lop_poly, rop_poly=rop_poly, rhp_poly=rhp_poly, alpha=alpha,
        ray_tol=op.ray_tol)
    df.to_csv(op.out_csv, index=False)

    if op.detail is not None:
        completeness_phenos.df_data_detail_.to_csv(op.detail, index=False)

import cip_python.nipype.interfaces.cip as cip
import cip_python.nipype.interfaces.unu as unu
import cip_python.nipype.interfaces.cip.cip_pythonWrap as cip_python_interfaces
from cip_python.nipype.cip_convention_manager import CIPConventionManager as CM
import nipype.pipeline.engine as pe         # the workflow and node wrappers
from nipype.pipeline.engine import Workflow
import tempfile, shutil
import pydot
import sys
import os 
import pdb

class VesselParticlesMaskWorkflow(Workflow):
    """This workflow produces a vessel seeds mask that is intended to be used
    as an input to the vessel particles routine. 

    Parameters
    ----------
    ct_file_name : str
        The file name of the CT image (single file, 3D volume) in which to 
        identify seeds as possible vessel locations.
    
    label_map_file_name : str
        File name for mask within which to idenfity vessel seeds.

    tmp_dir : str
        Directory in which to store intermediate files used for this computation
        
    vessel_seeds_mask_file_name : str, optional
        File name of output vessel seeds mask. If none is specified, a file name
        will be created using the CT file name prefix with the suffix
        _vesselSeedsMask.nhdr. The seeds mask indicates possible vessel loctions
        with the Vessel chest type label.
    """
    def __init__(self, ct_file_name, label_map_file_name, 
                 tmp_dir, vessel_seeds_mask_file_name=None):
        Workflow.__init__(self, 'VesselParticlesMaskWorkflow')

        assert ct_file_name.rfind('.') != -1, "Unrecognized CT file name format"
        
        self._tmp_dir = tmp_dir
        self._cid = ct_file_name[max([ct_file_name.rfind('/'), 0])+1:\
                                 ct_file_name.rfind('.')]

        if ct_file_name.rfind('/') != -1:
            self._dir = ct_file_name[0:ct_file_name.rfind('/')]
        else:
            self._dir = '.'

        if vessel_seeds_mask_file_name is None:
            self._vessel_seeds_mask_file_name = \
              os.path.join(self._dir, self._cid + CM._vesselSeedsMask)
        else:
            self._vessel_seeds_mask_file_name = vessel_seeds_mask_file_name
            
        # Params for feature strength computation
        self._ct_file_name = ct_file_name
        self._label_map_file_name = label_map_file_name
        self._distance_map_file_name = \
          os.path.join(self._tmp_dir, self._cid + '_distanceMap.nhdr') 
        self._feature_mask_file_name = \
          os.path.join(self._tmp_dir, self._cid + '_featureMask.nhdr')
        self._masked_strength_file_name = \
          os.path.join(self._tmp_dir, self._cid + '_maskedStrength.nhdr')
        self._equalized_strength_file_name = \
          os.path.join(self._tmp_dir, self._cid + '_equalized.nhdr')
        self._thresholded_equalized_file_name = \
          os.path.join(self._tmp_dir, self._cid + '_thresholded.nhdr')
        self._converted_thresholded_equalized_file_name = \
          os.path.join(self._tmp_dir, self._cid + '_converted.nhdr')
        self._strength_file_name = \
          os.path.join(self._tmp_dir, self._cid + '_strength.nhdr')
        self._scale_file_name = \
          os.path.join(self._tmp_dir, self._cid + '_scale.nhdr')
        self._thinned_file_name = \
          os.path.join(self._tmp_dir, self._cid + '_thinned.nhdr')          
        self._sigma_step_method = 1
        self._rescale = False
        self._threads = 0
        self._method = 'Frangi'
        self._alpha = 0.5 # In [0, 1]
        self._beta = 0.5 # In [0. 1]
        self._C = 250 # In [0, 300]
        self._alphase = 0.25 # In [0, 1]
        self._nu = 0 # In [-1, 0.5]
        self._kappa = 0.5 # In [0.01, 2]
        self._betase = 0.1 # In [0.01, 2]
        self._sigma = 1.0
        self._sigma_min = 0.7 
        self._sigma_max = 4.0
        self._num_steps = 4         
        self._gaussianStd = [self._sigma_min, self._sigma_max, self._num_steps]

        # Params for histogram equalization (unu heq node)
        self._bin = 10000
        self._amount = 0.5
        self._smart = 2

        # Param for thresholding the histogram-equalized strength image
        self._vesselness_th = 0.5
        
        # Params for 'unu_2op_lt' node
        self._distance_from_wall = -2.0

        # Create distance map node. We want to isolate a region that is 
        # not too close to the lung periphery (particles can pick up noise in
        # that region)
        compute_distance_transform = \
          pe.Node(interface=cip.ComputeDistanceMap(), 
                  name='compute_distance_transform')
        compute_distance_transform.inputs.labelMap = self._label_map_file_name
        compute_distance_transform.inputs.distanceMap = \
          self._distance_map_file_name
        
        # Create node for thresholding the distance map
        unu_2op_lt = pe.Node(interface=unu.unu_2op(), name='unu_2op_lt')
        unu_2op_lt.inputs.operator = 'lt'
        unu_2op_lt.inputs.type = 'short'
        unu_2op_lt.inputs.in2_scalar = self._distance_from_wall
        unu_2op_lt.inputs.output = self._feature_mask_file_name

        # Create node for generating the vesselness feature strength image
        compute_feature_strength = \
          pe.Node(interface=cip.ComputeFeatureStrength(),
                  name='compute_feature_strength')
        compute_feature_strength.inputs.inFileName = self._ct_file_name
        compute_feature_strength.inputs.outFileName = self._strength_file_name  
        compute_feature_strength.inputs.ssm = self._sigma_step_method
        compute_feature_strength.inputs.rescale = self._rescale
        compute_feature_strength.inputs.threads = self._threads
        compute_feature_strength.inputs.method = self._method
        compute_feature_strength.inputs.feature = 'RidgeLine'
        compute_feature_strength.inputs.alpha = self._alpha
        compute_feature_strength.inputs.beta = self._beta  
        compute_feature_strength.inputs.C = self._C
        compute_feature_strength.inputs.alphase = self._alphase
        compute_feature_strength.inputs.nu = self._nu
        compute_feature_strength.inputs.kappa = self._kappa
        compute_feature_strength.inputs.betase = self._betase
        compute_feature_strength.inputs.std = self._gaussianStd
        
        # Create node for isolating the strength values within the mask
        unu_2op_x_iso = pe.Node(interface=unu.unu_2op(), name='unu_2op_x_iso')
        unu_2op_x_iso.inputs.operator = 'x'
        unu_2op_x_iso.inputs.type = 'float'
        unu_2op_x_iso.inputs.output = self._masked_strength_file_name

        # Create node for histogram equalization
        unu_heq = pe.Node(interface=unu.unu_heq(), name='unu_heq')
        unu_heq.inputs.bin = self._bin
        unu_heq.inputs.amount = self._amount
        unu_heq.inputs.smart = self._smart
        unu_heq.inputs.output = self._equalized_strength_file_name

        # Set up thresholder to isolate all histogram-equalized voxels that
        # are above a pre-defined threshold
        unu_2op_gt = pe.Node(interface=unu.unu_2op(), name='unu_2op_gt')
        unu_2op_gt.inputs.operator = 'gt'
        unu_2op_gt.inputs.in2_scalar = self._vesselness_th = 0.5
        unu_2op_gt.inputs.output = self._thresholded_equalized_file_name

        # Now convert the equalized, thresholded image to short type
        unu_convert = pe.Node(interface=unu.unu_convert(), name='unu_convert')
        unu_convert.inputs.type = 'short'
        unu_convert.inputs.output = \
          self._converted_thresholded_equalized_file_name

        # Thin the thresholded image
        generate_binary_thinning_3d = \
          pe.Node(interface=cip.GenerateBinaryThinning3D(), 
                  name='generate_binary_thinning_3d')
        generate_binary_thinning_3d.inputs.out = \
          self._thinned_file_name

        # Create node for isolating the strength values within the mask
        unu_2op_x_ves = pe.Node(interface=unu.unu_2op(), name='unu_2op_x_ves')
        unu_2op_x_ves.inputs.operator = 'x'
        unu_2op_x_ves.inputs.output = self._vessel_seeds_mask_file_name
        unu_2op_x_ves.inputs.in1_scalar = 768
        
        # Set up the workflow connections        
        self.connect(compute_distance_transform, 'distanceMap', 
                     unu_2op_lt, 'in1_file')
        self.connect(unu_2op_lt, 'output', unu_2op_x_iso, 'in1_file')
        self.connect(compute_feature_strength, 'outFileName', 
                     unu_2op_x_iso, 'in2_file')
        self.connect(unu_2op_x_iso, 'output', unu_heq, 'input')        
        self.connect(unu_heq, 'output', unu_2op_gt, 'in1_file')        
        self.connect(unu_2op_gt, 'output', unu_convert, 'input')
        self.connect(unu_convert, 'output', 
                     generate_binary_thinning_3d, 'opt_in')
        self.connect(generate_binary_thinning_3d, 'out', 
                     unu_2op_x_ves, 'in2_file')                

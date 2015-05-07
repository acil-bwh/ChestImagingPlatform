import cip_python.nipype.interfaces.cip as cip
import cip_python.nipype.interfaces.unu as unu
import cip_python.nipype.interfaces.ITKTools as tools
import cip_python.nipype.interfaces.cip.cip_pythonWrap as cip_python_interfaces
import nipype.interfaces.spm as spm         # the spm interfaces
import nipype.pipeline.engine as pe         # the workflow and node wrappers
from nipype.pipeline.engine import Workflow
import pydot
import sys
import os 
from nipype import SelectFiles, Node
import pdb

class VesselParticlesMaskWorkflow(Workflow):
    """

    Parameters
    ----------
    ct_file_name : str
    
    method : string
        Must be either 'Frangi' or 'StrainEnergy'
    
    """
    def __init__(self):
        Workflow.__init__(self, 'VesselParticlesMaskWorkflow')
    
        # Params for feature strength computation
        self._ct_file_name = '/Users/jross/Downloads/ChestImagingPlatform/Testing/Data/Input/vessel.nrrd' 
        self._label_map_file_name = '/Users/jross/Downloads/acil/Experiments/tune_vessel_particles_mask/Data/volume_mask_eroded.nrrd'
        self._distance_map_file_name = '/Users/jross/Downloads/acil/Experiments/tune_vessel_particles_mask/Data/distance_map.nrrd'
        self._feature_mask_file_name = '/Users/jross/Downloads/acil/Experiments/tune_vessel_particles_mask/Data/feature_mask.nrrd'
        self._masked_strength_file_name = '/Users/jross/Downloads/acil/Experiments/tune_vessel_particles_mask/Data/masked_strength.nrrd'
        self._equalized_strength_file_name = '/Users/jross/Downloads/acil/Experiments/tune_vessel_particles_mask/Data/equalized_strength_file_name.nrrd'
        self._thresholded_equalized_file_name = '/Users/jross/Downloads/acil/Experiments/tune_vessel_particles_mask/Data/thresholded_equalized_file_name.nrrd'
        self._converted_thresholded_equalized_file_name = '/Users/jross/Downloads/acil/Experiments/tune_vessel_particles_mask/Data/converted_thresholded_equalized_file_name.nrrd'
        self._vessel_seeds_mask_file_name = '/Users/jross/Downloads/acil/Experiments/tune_vessel_particles_mask/Data/vessel_seeds_mask.nrrd'        
        self._strength_file_name = '/Users/jross/tmp/tmp_strength.nrrd'
        self._scale_file_name = '/Users/jross/tmp/tmp_scale.nrrd'
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
        compute_distance_transform.inputs.distanceMap = self._distance_map_file_name
        
        # Create node for thresholding the distance map
        unu_2op_lt = pe.Node(interface=unu.unu_2op(), name='unu_2op_lt')
        unu_2op_lt.inputs.operator = 'lt'
        unu_2op_lt.inputs.type = 'short'
        unu_2op_lt.inputs.in2_scalar = self._distance_from_wall
        unu_2op_lt.inputs.output = self._feature_mask_file_name

        #self.connect(px_distance_transform, 'out', unu_2op_lt, 'in1_file')
        self.connect(compute_distance_transform, 'distanceMap', unu_2op_lt, 'in1_file')        

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

        unu_2op_x = pe.Node(interface=unu.unu_2op(), name='unu_2op_x')
        unu_2op_x.inputs.operator = 'x'
        unu_2op_x.inputs.type = 'float'
        unu_2op_x.inputs.output = self._masked_strength_file_name

        self.connect(unu_2op_lt, 'output', unu_2op_x, 'in1_file')
        self.connect(compute_feature_strength, 'outFileName', unu_2op_x, 'in2_file')        

        # Create node for histogram equalization
        unu_heq = pe.Node(interface=unu.unu_heq(), name='unu_heq')
        unu_heq.inputs.bin = self._bin
        unu_heq.inputs.amount = self._amount
        unu_heq.inputs.smart = self._smart
        unu_heq.inputs.output = self._equalized_strength_file_name

        self.connect(unu_2op_x, 'output', unu_heq, 'input')

        # Set up thresholder to isolate all histogram-equalized voxels that
        # are above a pre-defined threshold
        unu_2op_gt = pe.Node(interface=unu.unu_2op(), name='unu_2op_gt')
        unu_2op_gt.inputs.operator = 'gt'
        unu_2op_gt.inputs.in2_scalar = self._vesselness_th = 0.5
        unu_2op_gt.inputs.output = self._thresholded_equalized_file_name
        
        self.connect(unu_heq, 'output', unu_2op_gt, 'in1_file')

        # Now convert the equalized, thresholded image to short type
        unu_convert = pe.Node(interface=unu.unu_convert(), name='unu_convert')
        unu_convert.inputs.type = 'short'
        unu_convert.inputs.output = self._converted_thresholded_equalized_file_name

        self.connect(unu_2op_gt, 'output', unu_convert, 'input')

        generate_binary_thinning_3d = \
          pe.Node(interface=cip.GenerateBinaryThinning3D(), 
                  name='generate_binary_thinning_3d')
        generate_binary_thinning_3d.inputs.out = self._vessel_seeds_mask_file_name

        self.connect(unu_convert, 'output', generate_binary_thinning_3d, 'opt_in')
        
    #    self.write_graph(dotfilename="/Users/jross/tmp/tmp.dot")    
    #    generate_binary_thinning_3d = \
    #      CIPNode(interface=cip.GenerateBinaryThinning3D(),
    #              name='generate_binary_thinning_3d')    
    #    self.add_nodes([generate_binary_thinning_3d])
    
               # Compute Frangi
    #                if self._init_method == 'Frangi':
    #                    tmpCommand = "ComputeFeatureStrength -i %(in)s -m Frangi -f RidgeLine --std %(minscale)f,4,%(maxscale)f --ssm 1 --alpha 0.5 --beta 0.5 --C 250 -o %(out)s"
    #                    tmpCommand = tmpCommand % {'in':ct_file_nameRegion,'out':featureMapFileNameRegion,'minscale':self._min_scale,'maxscale':self._max_scale}
    #                    
    #                    #Hist equalization, threshold Feature strength and masking
    #                    tmpCommand = "unu 2op x %(feat)s %(mask)s -t float | unu heq -b 10000 -a 0.5 -s 2 | unu 2op gt - %(vesselness_th)f  | unu convert -t short -o %(out)s"
    #                    tmpCommand = tmpCommand % {'feat':featureMapFileNameRegion,'mask':pl_file_nameRegion,'vesselness_th':self._vesselness_th,'out':maskFileNameRegion}
    #                elif self._init_method == 'StrainEnergy':
    #                    tmpCommand = "ComputeFeatureStrength -i %(in)s -m StrainEnergy -f RidgeLine --std %(minscale)f,4,%(maxscale)f --ssm 1 --alpha 0.2 --beta 0.1 --kappa 0.5 --nu 0.1 -o %(out)s"
    #                    tmpCommand = tmpCommand % {'in':ct_file_nameRegion,'out':featureMapFileNameRegion,'minscale':self._min_scale,'maxscale':self._max_scale}
    #                    tmpCommand  = os.path.join(path['CIP_PATH'],tmpCommand)
    #                    
    #                    #Hist equalization, threshold Feature strength and masking
    #                    tmpCommand = "unu 2op x %(feat)s %(mask)s -t float | unu heq -b 10000 -a 0.5 -s 2 | unu 2op gt - %(vesselness_th)f  | unu convert -t short -o %(out)s"
    #                    tmpCommand = tmpCommand % {'feat':featureMapFileNameRegion,'mask':pl_file_nameRegion,'vesselness_th':self._vesselness_th,'out':maskFileNameRegion}
    #                elif self._init_method == 'Threshold':
    #                    tmpCommand = "unu 2op gt %(in)s %(intensity_th)f | unu 2op x - %(mask)s -o %(out)s"
    #                    tmpCommand = tmpCommand % {'in':ct_file_nameRegion,'mask':pl_file_nameRegion,'intensity_th':self._intensity_th,'out':maskFileNameRegion}
    #                
    #                #Binary Thinning
    #                tmpCommand = "GenerateBinaryThinning3D -i %(in)s -o %(out)s"
    #                tmpCommand = tmpCommand % {'in':maskFileNameRegion,'out':maskFileNameRegion}
    

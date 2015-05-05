import cip_python.nipype.interfaces.cip as cip
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
    
        # Compute feature strength
        self._ct_file_name = '/Users/jross/Downloads/ChestImagingPlatform/Testing/Data/Input/vessel.nrrd' 
        self._strength_file_name = '/Users/jross/tmp/tmp_strength.nrrd'
        self._scale_file_name = '/Users/jross/tmp/tmp_scale.nrrd'
        self._sigma_step_method = 1
        self._rescale = False
        self._threads = 0
        self._method = 'Frangi'
        self._alpha = 0.5 # In [0, 1]
        self._beta = 0.5 # In [0. 1]
        self._C = 0.5 # In [0, 300]
        self._alphase = 0.25 # In [0, 1]
        self._nu = 0 # In [-1, 0.5]
        self._kappa = 0.5 # In [0.01, 2]
        self._betase = 0.1 # In [0.01, 2]

        self._sigma = 1.0
        self._sigma_min = 1.0 
        self._sigma_max = 2.0
        self._num_steps = 3
        
        #self._gaussianStd = [self._sigma, self._sigma_min, self._sigma_max, self._num_steps]
        self._gaussianStd = [self._sigma]        
        #Gaussian smoothing standard deviation. 1 value: sigma, 3 values: sigmaMin, sigmaMax, numberOfSteps

        compute_feature_strength = \
          pe.Node(interface=cip.ComputeFeatureStrength(),
                  name='compute_feature_strength')
        self.add_nodes([compute_feature_strength])
        self.get_node('compute_feature_strength').set_input('inFileName', self._ct_file_name)
        self.get_node('compute_feature_strength').set_input('outFileName', self._strength_file_name)    
        self.get_node('compute_feature_strength').set_input('outScaleFileName', self._scale_file_name)    
        self.get_node('compute_feature_strength').set_input('ssm', self._sigma_step_method)    
        self.get_node('compute_feature_strength').set_input('rescale', self._rescale)    
        self.get_node('compute_feature_strength').set_input('threads', self._threads)    
        self.get_node('compute_feature_strength').set_input('method', self._method)    
        self.get_node('compute_feature_strength').set_input('feature', 'RidgeLine')    
        self.get_node('compute_feature_strength').set_input('alpha', self._alpha)    
        self.get_node('compute_feature_strength').set_input('beta', self._beta)    
        self.get_node('compute_feature_strength').set_input('C', self._C)
        self.get_node('compute_feature_strength').set_input('alphase', self._alphase)                                        
        self.get_node('compute_feature_strength').set_input('nu', self._nu)
        self.get_node('compute_feature_strength').set_input('kappa', self._kappa)
        self.get_node('compute_feature_strength').set_input('betase', self._betase)
        self.get_node('compute_feature_strength').set_input('std', self._gaussianStd)    
        self.run()
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
    

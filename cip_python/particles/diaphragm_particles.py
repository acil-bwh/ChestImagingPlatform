#!/usr/bin/python

# TODO: Investigate live_thresh and seed_thresh (ranges specified below)
# TODO: Consider using mask throughout all passes if you're passing an
#       airway mask in (tried this -- does not seem to have an effect. Bug?)
# TODO: Investigate alpha and beta settings -- esp. for pass three. In some
#       passes they are irrelevant.

import os
from optparse import OptionParser

from cip_python.particles import ChestParticles

class DiaphragmParticles(ChestParticles):
    """Class for diaphragm-specific particles sampling

    Paramters
    ---------
    in_file_name : string
        File name of input volume

    out_particles_file_name : string
        File name of the output particles

    tmp_dir : string
        Name of temporary directory in which to store intermediate files

    mask_file_name : string (optional)
        File name of mask within which to execute particles

    max_scale : float (optional)
        The maximum scale to consider in scale space (default is 6.0). If
        larger structures are desired, it's advised to downsample the input
        image using the 'down_sample_rate' parameter and not to simply increase
        'max_scale'. For example, to capture a structure that is best
        represented at a scale of 12, keep 'max_scale' at 6 and downsample by
        2. The scale of the output particles is handled properly.    

    live_thresh : float (optional)
        Default is -100. Possible interval to explore: [-30, -150]

    seed_thresh : float (optional)
        Default is -80. Possible interval to explore: [-30, -200]

    scale_samples : int (optional)
        The number of pre-blurrings performed on the input image. These
        pre-blurrings are saved to the specified temp directory and used for
        interpolation across scale. The scale at which a given blurring is
        performed is also a function of the 'max_scale' parameter. Note that
        blurrings are not performed uniformly on the interval [0, max_scale].
        Instead, more blurrings are performed at the low end in order to better
        capture smaller structures. Default value is 5.

    down_sample_rate : int (optional)
        The amount by which the input image will be downsampled prior to
        running particles. Default is 1 (no downsampling).

    """
    def __init__(self, in_file_name, out_particles_file_name, tmp_dir,
                 mask_file_name=None, max_scale=3., live_thresh=-100.,
                 seed_thresh=-80., scale_samples=3, down_sample_rate=1, min_intensity=-200, max_intensity=200):
        ChestParticles.__init__(self, feature_type="ridge_surface",
                            in_file_name=in_file_name,
                            out_particles_file_name=out_particles_file_name,
                            tmp_dir=tmp_dir, mask_file_name=mask_file_name,
                            max_scale=max_scale, scale_samples=scale_samples,
                            down_sample_rate=down_sample_rate)
        self._max_intensity = max_intensity
        self._min_intensity = min_intensity
        self._live_thresh = live_thresh
        self._seed_thresh = seed_thresh
        self._max_scale = max_scale
        self._scale_samples = scale_samples
        self._down_sample_rate = down_sample_rate
        self._mode_thresh = -0.5
        self._population_control_period = 10
        self._no_add = 0
  
    def execute(self):
        #Pre-processing
        if self._down_sample_rate > 1:
            downsampled_vol = os.path.join(self._tmp_dir, "ct-down.nrrd")
            self.down_sample(self._in_file_name, downsampled_vol, \
                             "cubic:0,0.5",self._down_sample_rate)
            if self._use_mask == True:
            	downsampled_mask = os.path.join(self._tmp_dir, \
                                                "mask-down.nrrd")
            	self.down_sample(self._mask_file_name, \
                                 downsampled_mask, "cheap",self._down_sample_rate)
            	self._tmp_mask_file_name = downsampled_mask
            
        else:
            downsampled_vol = self._in_file_name
            self._tmp_mask_file_name = self._mask_file_name

        print ("2")
        deconvolved_vol = os.path.join(self._tmp_dir, "ct-deconv.nrrd")
        self.deconvolve(downsampled_vol, deconvolved_vol)
        print ("finished deconvolution\n")
              
        #Setting member variables that will not change
        self._tmp_in_file_name = deconvolved_vol
        
        # Temporary nrrd particles points
        out_particles = os.path.join(self._tmp_dir, "pass%d.nrrd")

        #Pass 1
        #Init params
        self._use_strength = False
        self._inter_particle_energy_type = "uni"
        self._init_mode = "PerVoxel"
        self._ppv = 1
        self._nss = 3

        # Energy
        # Radial energy function (psi_1 in the paper)
        self._inter_particle_enery_type = "uni"
        self._beta  = 0.7 # Irrelevant for pass 1
        self._alpha = 0.5
        self._irad = 4
        self._srad = 1.2
        self._iterations = 10

        #Build parameters and run
        print ("resetting param groups\n")
        self.reset_params()
        print ("building param groups\n")
        self.build_params()
        print ("Starting pass 1\n")
        self.execute_pass(out_particles % 1)
        print ("Finished pass 1\n")

        # Pass 2
        # Init params
        self._init_mode = "Particles"
        self._in_particles_file_name = out_particles % 1
        self._use_mask = False #TODO: was 0

        # Energy
        # Radial energy function (psi_2 in the paper).
        # Addition of 2 components: scale and space
        self._inter_particle_energy_type = "add"
        self._alpha = 0

        # Controls blending in scale and space with respect to
        # function psi_2
        self._beta = 0.5
        self._irad = 4
        self._srad = 2
        self._use_strength = True

        self._iterations = 20

        # Build parameters and run
        self.reset_params()
        self.build_params()
        print ("starting pass 2\n")
        self.execute_pass(out_particles % 2)
        print ("finished pass 2\n")

        # Pass 3
        self._init_mode = "Particles"
        self._in_particles_file_name = out_particles % 2
        self._use_mask = False # TODO: was 0

        # Energy
        self._inter_particle_energy_type = "add"
        self._alpha = 0.25
        self._beta = 0.25
        self._gamma = 0.002
        self._irad = 4
        self._srad = 4
        self._use_strength = True
        self._use_mode_th = True
        self._iterations = 50

        # Build parameters and run
        self.reset_params()
        self.build_params()
        print ("starting pass 3\n")
        self.execute_pass(out_particles % 3)
        print ("finished pass 3\n")

        # Probe quantities and save to VTK
        print ("about to probe\n")
        self.probe_quantities(self._tmp_in_file_name, out_particles % 3)
        print ("finished probing\n")
        #Adjust scale if down-sampling was performed
        if self._down_sample_rate > 1:
            self.adjust_scale(out_particles % 3)
        print ("about to save to vtk\n")
        self.save_vtk(out_particles % 3)
        print ("finished saving\#####n")

        #Clean tmp Directory
        self.clean_tmp_dir()


if __name__ == "__main__":

  parser = OptionParser()
  parser.add_option("-i", help='input CT scan', dest="input_ct")
  parser.add_option("-m", help='input mask for seeding', dest="input_mask")
  parser.add_option("-o", help='output particles (vtk format)', dest="output_particles")
  parser.add_option("-t", help='tmp directory', dest="tmp_dir")
  parser.add_option("-s", help='max scale', dest="max_scale",default=3.0)
  parser.add_option("-r", help='down sampling rate (>=1)', dest="down_sample_rate",default=1.0)
  parser.add_option("-n", help='number of scale volumes', dest="scale_samples",default=3)
  parser.add_option("--lth", help='live threshold (<0)', dest="live_th",default=-100)
  parser.add_option("--sth", help='seed threshold (<0)', dest="seed_th",default=-80)
  parser.add_option("--minI", help='min intensity for feature', dest="min_intensity",default=-200)
  parser.add_option("--maxI", help='max intensity for feature', dest="max_intensity",default=200)

  (op, args) = parser.parse_args()

  #Max scale has to be an integer for multi-scale particles
  if int(op.scale_samples) > 1:
    max_scale = round(float(op.max_scale))
  else:
    max_scale = float(op.max_scale)

  dp=DiaphragmParticles(op.input_ct,op.output_particles,op.tmp_dir,op.input_mask,max_scale,\
                        float(op.live_th),float(op.seed_th),int(op.scale_samples),float(op.down_sample_rate),\
                        float(op.min_intensity),float(op.max_intensity))
  dp.execute()


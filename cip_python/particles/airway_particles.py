#!/usr/bin/python

# TODO: Investigate live_thresh and seed_thresh (ranges specified below)
# TODO: Consider using mask throughout all passes if you're passing an
#       airway mask in (tried this -- does not seem to have an effect. Bug?)
# TODO: Investigate alpha and beta settings -- esp. for pass three. In some
#       passes they are irrelevant.

import os
import tempfile, shutil
from cip_python.particles import ChestParticles

class AirwayParticles(ChestParticles):
    """Class for airway-specific particles sampling

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
        Default is 40. Possible interval to explore: [10, 150]

    seed_thresh : float (optional)
        Default is 30. Possible interval to explore: [10, 200]

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
                 mask_file_name=None, max_scale=6., live_thresh=135.75,
                 seed_thresh=125.55, scale_samples=6, down_sample_rate=1,
                 min_intensity=-1100, max_intensity=-500):
        ChestParticles.__init__(self, feature_type="valley_line",
                            in_file_name=in_file_name,
                            out_particles_file_name=out_particles_file_name,
                            tmp_dir=tmp_dir, mask_file_name=mask_file_name,
                            max_scale=max_scale, scale_samples=scale_samples,
                            down_sample_rate=down_sample_rate)
        self._max_intensity = max_intensity
        self._min_intensity = min_intensity
        self._max_scale = max_scale
        self._scale_samples = scale_samples
        self._down_sample_rate = down_sample_rate
          
        self._live_thresh = live_thresh
        self._seed_thresh = seed_thresh
       
        self._phase_iterations = [48,180,40]

        self._phase_irads=[1.94396,1.52298,1.0]
        self._phase_srads=[1.05966,4.12374,4.94739]

        self._phase_population_control_periods = [3,9,18]
        self._phase_alphas = [0.990458, 0.999429, 0.694357]
        self._phase_betas = [0.976696, 0.228277, 0.999034]
        self._phase_gammas = [0.107816, 0.367268, 0.51446]

        self._cip_type = 'Airway'
        
        #Default init
        self._init_mode = "PerVoxel"
        self._ppv = 1
        self._nss = 2
    
        self._binning_width = 1.5

    def execute(self):

        self.preprocessing()
              
        # Temporary nrrd particles points
        out_particles = os.path.join(self._tmp_dir, "pass%d.nrrd")
              
        #Pass 1
        #Init params
        self._use_strength = False
        self._inter_particle_energy_type = "uni"

        # Energy
        # Radial energy function (psi_1 in the paper)
        self._inter_particle_enery_type = "uni"
        self._alpha = self._phase_alphas[0]
        self._beta  = self._phase_betas[0] # Irrelevant for pass 1
        self._gamma = self._phase_gammas[0]
        self._irad = self._phase_irads[0]
        self._srad = self._phase_srads[0]
        self._iterations = self._phase_iterations[0]
        self._population_control_period = self._phase_population_control_periods[0]

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
        self._use_mask = False

        # Energy
        # Radial energy function (psi_2 in the paper).
        # Addition of 2 components: scale and space
        self._inter_particle_energy_type = "add"
        self._alpha = self._phase_alphas[1]
        # Controls blending in scale and space with respect to
        # function psi_2
        self._beta = self._phase_betas[1]
        self._gamma = self._phase_gammas[1]
        self._irad = self._phase_irads[1]
        self._srad = self._phase_srads[1]
        self._use_strength = True

        self._iterations = self._phase_iterations[1]
        self._population_control_period = self._phase_population_control_periods[1]

        # Build parameters and run
        self.reset_params()
        self.build_params()
        print ("starting pass 2\n")
        self.execute_pass(out_particles % 2)
        print ("finished pass 2\n")

        # Pass 3
        self._init_mode = "Particles"
        self._in_particles_file_name = out_particles % 2
        self._use_mask = False

        # Energy
        self._inter_particle_energy_type = "add"
        self._alpha = self._phase_alphas[2]
        self._beta = self._phase_betas[2]
        self._gamma = self._phase_gammas[2]
        self._irad = self._phase_irads[2]
        self._srad = self._phase_srads[2]
        self._use_strength = True

        self._iterations = self._phase_iterations[2]
        self._population_control_period = self._phase_population_control_periods[2]

        # Build parameters and run
        self.reset_params()
        self.build_params()
        print ("starting pass 3\n")
        self.execute_pass(out_particles % 3)
        print ("finished pass 3\n")

        # Probe quantities and save to VTK
        print ("about to probe\n")
        self.probe_quantities(self._sp_in_file_name, out_particles % 3)
        print ("finished probing\n")

        print ("Saving to vtk...")
        self.save_vtk(out_particles % 3)
        print ("finished saving\#####n")

        #Clean tmp Directory
#        self.clean_tmp_dir()

if __name__ == "__main__":
  from argparse import ArgumentParser

  parser = ArgumentParser(description='Airway particles generation tool.')
  
  parser.add_argument("-i", help='input CT scan', dest="input_ct",required=True)
  parser.add_argument("-m", help='input mask for seeding', dest="input_mask",
    default=None)
  parser.add_argument("-p", help='input particle points to initialize (if not \
    specified a per-voxel approach is used)', dest="input_particles", 
    default=None)
  parser.add_argument("-o", help='output particles (vtk format)',
                    dest="output_particles",required=True)
  parser.add_argument("-t", help='tmp directory. if not provided, it creates a tmp system dir', dest="tmp_dir")
  parser.add_argument("-s", help='max scale [default: %(default)s)]',
                    dest="max_scale", default=6.0, type=float)
  parser.add_argument("-r", help='down sampling rate (>=1) [default: \
    %(default)s]', dest="down_sample_rate", default=1.0, type=float)
  parser.add_argument("-n", help='number of scale volumes [default: \
    %(default)s]', dest="scale_samples", default=6, type=int)
  parser.add_argument("--lth", help='live threshold (>0) [default: \
    %(default)s]', dest="live_th", default=135.705, type=float)
  parser.add_argument("--sth", help='seed threshold (>0) [default: \
    %(default)s]', dest="seed_th", default=125.155, type=float)
  parser.add_argument("--minI",
    help='min intensity for feature [default: %(default)s]',
    dest="min_intensity", default=-1100, type=float)
  parser.add_argument("--maxI",
    help='max intensity for feature [default: %(default)s]',
    dest="max_intensity", default=-500, type=float)
  parser.add_argument("--perm",dest="permissive",action='store_true',
    help='Permissive mode enable. Allow volumes of different shapes and origin')

  op = parser.parse_args()

  if op.tmp_dir is not None:
      tmp_dir = op.tmp_dir
  else:        
      tmp_dir = tempfile.mkdtemp()
  
  ap = AirwayParticles(op.input_ct, op.output_particles, tmp_dir,
    op.input_mask, float(op.max_scale), float(op.live_th), float(op.seed_th), 
    int(op.scale_samples), float(op.down_sample_rate), float(op.min_intensity),
    float(op.max_intensity))

  if op.permissive == True:
    ap._permissive=True

  if op.input_particles == None:
      pass
  else:
      ap._init_mode="Particles"
      ap._in_particles_file_name = op.input_particles
  
  ap.execute()

  if op.tmp_dir is None:
      shutil.rmtree(tmp_dir)

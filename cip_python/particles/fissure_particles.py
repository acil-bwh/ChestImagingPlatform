from argparse import ArgumentParser
import tempfile, shutil
import os

from cip_python.particles import ChestParticles

class FissureParticles(ChestParticles):
    """Class for fissure-specific particles sampling

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

    scale : float (optional)
        The scale of the fissure to consider in scale space.

    live_thresh : float (optional)
        Threshold to use when pruning particles during population control.
    
    seed_thresh : float (optional)
        Threshold to use when initializing particles.

    down_sample_rate : int (optional)
        The amount by which the input image will be downsampled prior to
        running particles. Default is 1 (no downsampling).

    min_intensity : int (optional)
        Histogram equilization will be applied to enhance the fissure features
        within the range [min_intensity, max_intensity].

    max_intensity : int (optional)
        Histogram equilization will be applied to enhance the fissure features
        within the range [min_intensity, max_intensity].        

    iterations : int (optional)
        The number of iterations for which to run the algorithm.

    perm : bool (optional)
        If true, allow mask and CT volumes to have different shapes or meta data
    """
    def __init__(self, in_file_name, out_particles_file_name, tmp_dir,
                 mask_file_name=None, scale=0.9, live_thresh=-15,
                 seed_thresh=-45, down_sample_rate=1,
                 min_intensity=-920, max_intensity=-400, iterations=100,init_mode="Random", 
                 perm=False):
        ChestParticles.__init__(self, feature_type="ridge_surface",
            in_file_name=in_file_name, 
            out_particles_file_name=out_particles_file_name,
            tmp_dir=tmp_dir, mask_file_name=mask_file_name,
            max_scale=scale, scale_samples=1,
            down_sample_rate=down_sample_rate)

        self._max_intensity = max_intensity
        self._min_intensity = min_intensity
        self._live_thresh = live_thresh
        self._seed_thresh = seed_thresh
        self._max_scale = scale
        self._scale_samples = 1
        self._down_sample_rate = down_sample_rate
        self._iterations = iterations        
        self._mode_thresh = -0.5
        self._population_control_period = 10
        self._no_add = 0
        self._cip_type = 'Fissure'
        #Setting up initialization
        print(init_mode)
        self._init_mode = init_mode
        #Config for PerVoxel init mode
        self._ppv = 1
        self._nss = 1
        #Config for Random init mode
        self._number_init_particles = 10000


        self._verbose = 1
        if perm:
            self._permissive = True
 
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

        deconvolved_vol = os.path.join(self._tmp_dir, "ct-deconv.nrrd")
        self.deconvolve(downsampled_vol, deconvolved_vol)

        #Setting member variables that will not change
        self._tmp_in_file_name = deconvolved_vol
        
        # Temporary nrrd particles points
        out_particles = os.path.join(self._tmp_dir, "pass%d.nrrd")

        self._single_scale = 1
        #Setting up single scale approach
        self._use_strength = False
        self._inter_particle_energy_type = "justr"

        self._beta  = 0 # Irrelevant for pass 1
        self._alpha = 0.5
        self._irad = 1.7 
        self._srad = 1.2
            
        #Build parameters and run
        self.reset_params()
        self.build_params()

        self.execute_pass(out_particles % 3)

        # Probe quantities and save to VTK
        self.probe_quantities(self._tmp_in_file_name, out_particles % 3)
        
        #Adjust scale if down-sampling was performed
        if self._down_sample_rate > 1:
            self.adjust_scale(out_particles % 3)
        self.save_vtk(out_particles % 3)

        #Clean tmp Directory
        self.clean_tmp_dir()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--ict", help='Input CT scan', dest="ict")
    parser.add_argument("--ilm", help='Input mask for seeding', 
      dest="ilm")
    parser.add_argument("--op", help='Output particles (vtk format)', \
      dest="op")
    parser.add_argument("--tmp", help='Temp directory in which to store \
      intermediate files', dest="tmp", default=None)
    parser.add_argument("--scale", help='The scale of the fissure to consider \
      in scale space.', dest="scale", default=0.9)
    parser.add_argument("--rate", help='Down sampling rate. Must be greater than \
      or equal to 1.0 (default 1.0)', dest="rate", default=1.0)
    parser.add_argument("--lth", help='Threshold to use when pruning particles \
      during population control. Must be less than zero', 
      dest="lth", default=-15)
    parser.add_argument("--sth", help='Threshold to use when initializing \
      particles. Must be less than zero', dest="sth", default=-45)
    parser.add_argument("--min_int", help='Histogram equilization will be applied \
      to enhance the fissure features within the range [min_int, max_int]', 
      dest="min_int", default=-920)
    parser.add_argument("--max_int", help='Histogram equilization will be applied \
      to enhance the fissure features within the range [min_int, max_int]', 
      dest="max_int", default=-400)
    parser.add_argument("--iters", help='Number of algorithm iterations \
      (default 100)', dest="iters", default=100) 
    parser.add_argument("--per_voxel_init", help='Inititalize seeding one particle \
        per voxel in the input seeding mask', dest="per_voxel_init",action='store_true')   
    parser.add_argument("--perm", help='Allow mask and CT volumes to have \
      different shapes or meta data', dest="perm", action='store_true')          

    op = parser.parse_args()
    
    if op.tmp is not None:
        tmp = op.tmp
    else:        
        tmp = tempfile.mkdtemp()

    if op.per_voxel_init:
       init_mode="PerVoxel"
    else:
       init_mode="Random"

    dp = FissureParticles(op.ict, op.op, tmp, op.ilm, float(op.scale), 
        float(op.lth), float(op.sth), float(op.rate), float(op.min_int), 
        float(op.max_int), int(op.iters), init_mode, op.perm)
    dp.execute()
    
    if op.tmp is None:
        shutil.rmtree(tmp)

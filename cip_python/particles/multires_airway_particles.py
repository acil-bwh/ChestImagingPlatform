#!/usr/bin/python

# TODO: Investigate live_thresh and seed_thresh (ranges specified below)
# TODO: Consider using mask throughout all passes if you're passing an
#       airway mask in (tried this -- does not seem to have an effect. Bug?)
# TODO: Investigate alpha and beta settings -- esp. for pass three. In some
#       passes they are irrelevant.

import os
import math
from optparse import OptionParser
from cip_python.particles import ChestParticles

class MultiResAirwayParticles(ChestParticles):
    """Class for multiresolution airway-specific particles sampling

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
        Default is 50. Possible interval to explore: [30, 150]

    seed_thresh : float (optional)
        Default is 40. Possible interval to explore: [30, 200]

    scale_samples : int (optional)
        The number of pre-blurrings performed on the input image. These
        pre-blurrings are saved to the specified temp directory and used for
        interpolation across scale. The scale at which a given blurring is
        performed is also a function of the 'max_scale' parameter. Note that
        blurrings are not performed uniformly on the interval [0, max_scale].
        Instead, more blurrings are performed at the low end in order to better
        capture smaller structures. Default value is 5.

    multi_res_levels : int (optional)
        Number of multi-resolution levels for the image decomposition. Default is 2.

    """
    def __init__(self, in_file_name, out_particles_file_name, tmp_dir,
                 mask_file_name=None, max_scale=8., live_thresh=45.,
                 seed_thresh=45., scale_samples=5, multi_res_levels=2):
        ChestParticles.__init__(self, feature_type="valley_line",
                            in_file_name=in_file_name,
                            out_particles_file_name=out_particles_file_name,
                            tmp_dir=tmp_dir, mask_file_name=mask_file_name,
                            max_scale=max_scale, scale_samples=scale_samples)
        self._multi_res_levels = multi_res_levels
        self._max_intensity = -400
        self._min_intensity = -1100
        self._live_thresh = live_thresh
        self._seed_thresh = seed_thresh
     
    def execute(self):            
        #Compute Resolution Pyramid
        max_scale_per_level = int(math.ceil(self._max_scale / 2**(self._multi_res_levels-1)))
        max_scale = self._max_scale
        mask_file_name = self._mask_file_name
        #Run particles for each level
        output_particles_list = list()
        for res_level in range(self._multi_res_levels,0,-1):
            self._down_sample_rate = 2**(res_level-1)
            self._max_scale = max_scale_per_level
            particles_per_level=self.execute_airway_level( res_level )
            output_particles_list.append(particles_per_level)

        #Merge particles and run final step for uniform redistribution along all the scales
        merged_particles = os.path.join(self._tmp_dir, "merged-particles.nrrd")
        self.merge_particles(output_particles_list,merged_particles)

        #Fix max scale & downsampling rate and perform final point redistribution
        #Deconvolution is not necessary because is in the cache from the last pass in the
        #resolution pyramid
        self._max_scale = max_scale
        self._tmp_in_file_name = os.path.join(self._tmp_dir, "ct-deconv.nrrd")

        self._init_mode = "Particles"
        self._in_particles_file_name = merged_particles
        self._use_mask = True # TODO: was 0
            
        # Energy
        self._inter_particle_energy_type = "add"
        self._alpha = 0.5
        self._beta = 0.5
        self._gamma = 0.000002
        self._irad = 1.15
        self._srad = 4
        self._use_strength = True
            
        self._iterations = 50
            
        # Build parameters and run
        self.reset_params()
        self.build_params()
        print ("starting final pass\n")
        self.execute_pass(merged_particles)
        print ("finished final pass\n")

        # Probe quantities and save to VTK
        print ("about to probe\n")
        self.probe_quantities(self._tmp_in_file_name, merged_particles)
        print ("finished probing\n")
        print ("about to save to vtk\n")
        self.save_vtk(merged_particles)
        print ("finished saving\#####n")
  
        #Clean tmp Directory
        self.clean_tmp_dir()

    def execute_airway_level (self,level):
        #Pre-processing
        if self._down_sample_rate > 1:
            downsampled_vol = os.path.join(self._tmp_dir, "ct-down.nrrd")
            self.down_sample(self._in_file_name, downsampled_vol, \
                             "cubic:0,0.5",self._down_sample_rate)
            if self._use_mask == True:
                downsampled_mask = os.path.join(self._tmp_dir, \
                                                "mask-down.nrrd")
                if level == self._multi_res_levels:
                    #First level of pyramid. Just downsample the original mask
                    self.down_sample(self._mask_file_name, \
                                 downsampled_mask, "cheap",self._down_sample_rate)
 
                else:
                    #Compute differential mask between resolution levels to speed up initial seeding
                    self.differential_mask(self._down_sample_rate,2*self._down_sample_rate,downsampled_mask)
            
                self._tmp_mask_file_name = downsampled_mask
        
        else:
            downsampled_vol = self._in_file_name
            if self._multi_res_levels > 1:
                downsampled_mask = os.path.join(self._tmp_dir, \
                                            "mask-down.nrrd")
                self.differential_mask(self._down_sample_rate,2*self._down_sample_rate,downsampled_mask)
                self._tmp_mask_file_name = downsampled_mask
            else:
                self._tmp_mask_file_name = self._mask_file_name

        deconvolved_vol = os.path.join(self._tmp_dir, "ct-deconv.nrrd")
        self.deconvolve(downsampled_vol, deconvolved_vol)
        print ("finished deconvolution\n")
        #Adjust seeding threshold levels to account for downsampling rate
        orig_live_thresh = self._live_thresh
        orig_seed_thresh = self._seed_thresh
        self._live_thresh = self._live_thresh/self._down_sample_rate
        self._seed_thresh = self._seed_thresh/self._down_sample_rate
        print ("level "+str(level)+"seed th: "+str(self._seed_thresh)+"live th: "+str(self._live_thresh))
        #Setting member variables that will not change
        self._tmp_in_file_name = deconvolved_vol
                  
        # Temporary nrrd particles points
        out_particles = os.path.join(self._tmp_dir, "pass%d-l%d.nrrd")
        #Pass 1
        #Init params
        self._use_strength = False
        self._inter_particle_energy_type = "uni"
        self._init_mode = "PerVoxel"
        self._ppv = 1
        self._nss = 2
        
        # Energy
        # Radial energy function (psi_1 in the paper)
        self._inter_particle_enery_type = "uni"
        self._beta  = 0.7 # Irrelevant for pass 1
        self._alpha = 1.0
        self._iterations = 10
        
        #Build parameters and run
        self.reset_params()
        self.build_params()
        print ("Starting pass 1\n")
        self.execute_pass(out_particles % (1,level))
        print ("Finished pass 1\n")
        
        # Pass 2
        # Init params
        self._init_mode = "Particles"
        self._in_particles_file_name = out_particles % (1,level)
        self._use_mask = True #TODO: was 0
        
        # Energy
        # Radial energy function (psi_2 in the paper).
        # Addition of 2 components: scale and space
        self._inter_particle_energy_type = "add"
        self._alpha = 0
        
        # Controls blending in scale and space with respect to
        # function psi_2
        self._beta = 0.5
        self._irad = 1.15
        self._srad = 4
        self._use_strength = True
        
        self._iterations = 20
        
        # Build parameters and run
        self.reset_params()
        self.build_params()
        print ("starting pass 2\n")
        self.execute_pass(out_particles % (2,level))
        print ("finished pass 2\n")
        
        # Pass 3
        self._init_mode = "Particles"
        self._in_particles_file_name = out_particles % (2,level)
        self._use_mask = True # TODO: was 0
        
        # Energy
        self._inter_particle_energy_type = "add"
        self._alpha = 0.5
        self._beta = 0.5
        self._gamma = 0.000002
        self._irad = 1.15
        self._srad = 4
        self._use_strength = True
        
        self._iterations = 50
        
        # Build parameters and run
        self.reset_params()
        self.build_params()
        print ("starting pass 3\n")
        self.execute_pass(out_particles % (3,level))
        print ("finished pass 3\n")
        
        # Adjust scale to account for the level of resolution
        self.adjust_scale(out_particles % (3,level))
        
        # Recover threshold
        self._live_thresh = orig_live_thresh
        self._seed_thresh = orig_seed_thresh
                  
        return out_particles % (3,level)

if __name__ == "__main__":
    desc = """Multi-resolution scale-space particles for airway segmentation."""
    
    parser = OptionParser(description=desc)
    parser.add_option('--ct_file', 
                      help='The CT file on which to run particles',
                      dest='ct_file', metavar='<string>',
                      default=None)
    parser.add_option('--out_file', 
                      help='The output particles file name (with .vtk \
                      extension)', dest='out_file', metavar='<string>',
                      default=None)
    parser.add_option('--mask_file', 
                      help='Mask file name. Particles execution will be \
                      restricted to the region in this mask',
                      dest='mask_file', metavar='<string>', default=None)
    parser.add_option('--tmp_dir', 
                      help='The temporary directory in which to store \
                      intermediate files', dest='tmp_dir', metavar='<string>',
                      default='/var/tmp/')
    parser.add_option('--live_thresh',  help='The live threshold',
                      dest='live_thresh', metavar='<float>', default=45.)
    parser.add_option('--seed_thresh',  help='The seed threshold',
                      dest='seed_thresh', metavar='<float>', default=45.)
    parser.add_option('--max_scale',  help='Max scale to consider',
                      dest='max_scale', metavar='<float>', default=10.)
    parser.add_option('--levels',  help='Number of resolution levels to use',
                      dest='levels', metavar='<int>', default=2)            

    (options, args) = parser.parse_args()

    if options.ct_file is None:
        raise ValueError("Must specify a CT file")
    if options.out_file is None:
        raise ValueError("Must specify an output file")

    particles = MultiResAirwayParticles(options.ct_file, options.out_file,
                                        options.tmp_dir, options.mask_file,
                                        live_thresh=float(options.live_thresh),
                                        seed_thresh=float(options.seed_thresh),
                                        max_scale=float(options.max_scale),
                                        multi_res_levels=int(options.levels))
    particles.execute()

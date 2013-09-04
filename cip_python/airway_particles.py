#!/usr/bin/python

# TODO: Investigate live_thresh and seed_thresh (ranges specified below)
# TODO: Consider using mask throughout all passes if you're passing an
#       airway mask in
# TODO: Investigate alpha and beta settings -- esp. for pass three. In some
#       passes they are irrelevant.
# TODO: Try setting max_intensity to -400 instead of -600

import os
import pdb
from cip_python.chest_particles import ChestParticles

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
        The maximum scale to consider in scale space (default is 6.0)

    live_thresh : float (optional)
        Default is 50. Possible interval to explore: [30, 150]

    seed_thresh : float (optional)
        Default is 40. Possible interval to explore: [30, 200]

    scale_samples : int (optional)
        Default is 5.    
    """
    def __init__(self, in_file_name, out_particles_file_name, tmp_dir,
                 mask_file_name=None, max_scale=6., live_thresh=50.,
                 seed_thresh=40., scale_samples=5):
        ChestParticles.__init__(self, feature_type="valley_line",
                            in_file_name=in_file_name,
                            out_particles_file_name=out_particles_file_name,
                            tmp_dir=tmp_dir, mask_file_name=mask_file_name,
                            max_scale=max_scale, scale_samples=scale_samples)
        self._max_intensity = -600 # -400?
        self._min_intensity = -1100
        self._live_thresh = live_thresh
        self._seed_thresh = seed_thresh        

    def execute(self):
        #Pre-processing
        if self._down_sample_rate > 1:
            downsampled_vol = os.path.join(self._tmp_dir, "ct-down.nrrd")
            self.down_sample(self._in_file_name, downsampled_vol, \
                             "cubic:0,0.5")
            if self._use_mask == 1:
            	downsampled_mask = os.path.join(self._tmp_dir, \
                                                "mask-down.nrrd")
            	self.down_sample(self._mask_file_name, \
                                 downsampled_mask, "cheap")
            	self._mask_file_name = downsampled_mask
            
        else:
            downsampled_vol = self._in_file_name

        print "2"
        deconvolved_vol = os.path.join(self._tmp_dir, "ct-deconv.nrrd")
        self.deconvolve(downsampled_vol, deconvolved_vol)
        print "finished deconvolution\n"
        print "loc1\n"
        #Setting member variables that will not change
        self._tmp_in_file_name = deconvolved_vol
        print "loc2\n"
        
        # Temporary nrrd particles points
        out_particles = os.path.join(self._tmp_dir, "pass%d.nrrd")
        print "loc3\n"
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
        print "resetting param groups\n"
        self.reset_params()
        print "building param groups\n"
        self.build_params()
        print "Starting pass 1\n"
        self.execute_pass(out_particles % 1)
        print "Finished pass 1\n"

        # Pass 2
        # Init params
        self._init_mode = "Particles"
        self._in_particles_file_name = out_particles % 1
        self._use_mask = 1 #TODO: was 0

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
        print "starting pass 2\n"
        self.execute_pass(out_particles % 2)
        print "finished pass 2\n"

        # Pass 3
        self._init_mode = "Particles"
        self._in_particles_file_name = out_particles % 2
        self._use_mask = 1 # TODO: was 0

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
        print "starting pass 3\n"
        self.execute_pass(out_particles % 3)
        print "finished pass 3\n"

        # Probe quantities and save to VTK
        print "about to probe\n"
        self.probe_quantities(self._tmp_in_file_name, out_particles % 3)
        print "finished probing\n"
        print "about to save to vtk\n"
        self.save_vtk(out_particles % 3)
        print "finished saving\#####n"


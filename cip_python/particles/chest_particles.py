#------------------------------------------------------------------------------------
# TODO: We will want to put some of the functionality into a base class from which
# 'Generate[Airway,Vessel,Fissure]Particles' can inherit.
#
# TODO: Create a separate class for probing points (post-processing unu calls)
#------------------------------------------------------------------------------------

import subprocess
import os
from ..utils import ReadNRRDsWriteVTK
from ..input_output import ImageReaderWriter

class ChestParticles:
    """Base class for airway, vessel, and fissure particles classes.

    Parameters
    ----------
    feature_type : string
        Takes on one or four values: "ridge_line" (for vessels), "valley_line"
        (for airways), "ridge_surface" (for fissure), "valley_surface"

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
    def __init__(self, feature_type, in_file_name, out_particles_file_name,
        tmp_dir, mask_file_name=None, max_scale=6.0, scale_samples=5,
        down_sample_rate=1):
        
        assert feature_type == "ridge_line" or feature_type == "valley_line" \
            or feature_type == "ridge_surface" or feature_type == \
            "valley_surface", "Invalid feature type"
        
        self._feature_type = feature_type        
        self._inverse_kernel_params = ""
        self._init_params = ""
        self._tmp_dir = os.path.splitdrive(tmp_dir)[1] # To avoid puller confusion on Windows
        self._in_file_name = os.path.splitdrive(in_file_name)[1]
        self._mask_file_name = os.path.splitdrive(mask_file_name)[1]
        self._in_particles_file_name = "NA"
        self._out_particles_file_name = os.path.splitdrive(out_particles_file_name)[1]
        self._cip_region = 'UndefinedRegion'
        self._cip_type = 'UndefinedType'
               
        # The following takes on values 'DiscreteGaussian' or
        # 'ContinuousGaussian' and controls the way in which spatial smoothing
        # is done.
        self._blurring_kernel_type = "DiscreteGaussian"

        # If set to true, the temporary directory will be wiped clean
        # after execution
        self._clean_tmp_dir = False

        # Scale-particles filenames:
        # -------------------
        # Volume name that is going to be used for scale-space particles.
        # This volume is the result of any pre-processing
        self._sp_in_file_name = \
          os.path.join(self._tmp_dir, os.path.split(self._in_file_name)[1])        
        self._sp_mask_file_name = self._mask_file_name
        # Nrrd file to contain particle data.
        self._sp_particles_file_name  = "nrrdParticlesFileName.nrrd"
              
        # Debug params:
        # ----------------
        self._verbose = 0 #Verbosity level for puller
        self._debug = False #Print class commands

        # Particle system setup params:
        # -----------------------------
        # Options: Bspline3, Bspline5, Bspline7 and C4:
        self._recon_kernel_type    = "C4"
        # Options: Random, Halton, Particles, PerVoxel:
        self._init_mode = "Random"
        # Run at a single scale vs a full scale-space exploration:
        if self._mask_file_name is not None:
            self._use_mask = True
        else:
            self._use_mask = False

        self._single_scale = 0
        self._max_scale = max_scale
        self._scale_samples = scale_samples
        self._iterations = 50

        self._permissive = False # Allow volumes to have different 
                                 # shapes (false is safer)

        # Energy specific params:
        # -----------------------
        self._irad  = 1.7
        self._srad  = 1.2
        self._alpha = 0.5
        self._beta  = 0.5
        self._gamma = 1
        # Options: justr, add:
        self._inter_particle_energy_type = "justr" 
        self._use_strength = "true"

        # Initialization params
        # -----------------
        self._ppv = 1 # Points per voxel.
        self._nss = 1 # Number of samples along scale axis
        self._jit = 1 # Jittering to do for each point
        self._number_init_particles = 10000 #Number of initial particles (used with Rnadom and Halton)
              
        # Optimizer params:
        # -----------------
        # Binning width as mulitple of irad. Increase this value run into
        # overflow of maximum number of bins.
        self._binning_width = 1.3 
        # Population control period
        self._population_control_period = 5
        # If enabled, it does not add points during population control
        self._no_add = 0 

        # Pre proc Parameters:
        # --------------------
        # Limit the uppe range of the input image
        self._max_intensity =  400
        # Limit the lower range of the input image
        self._min_intensity = -1024
        # Downsampling factor to enable multiresolution particles
        self._down_sample_rate = down_sample_rate

        # Basic contrast Parameters
        # -------------------------
        self._live_thresh = -150 # Threshold on feature strength
        self._seed_thresh = -100 # Threshold on feature strengh
        self._mode_thresh = -0.5 # Threshold on mode of the hessian
        self._use_mode_thresh = False #Use mode threshold
  
        # Probing quantities
        #--------------------------
        # Dictionary with a list containing the vtk array name and a flag indicating if the
        # the quantity is going to used normalized derivatives.
        self._probing_quantities = dict()
        self._probing_quantities["val"]=["val",0]
        self._probing_quantities["heval0"]=["h0",1]
        self._probing_quantities["heval1"]=["h1",1]
        self._probing_quantities["heval2"]=["h2",1]
        self._probing_quantities["hmode"]=["hmode",1]
        self._probing_quantities["hevec0"]=["hevec0",0]
        self._probing_quantities["hevec1"]=["hevec1",0]
        self._probing_quantities["hevec2"]=["hevec2",0]
        self._probing_quantities["hess"]=["hess",0]
      
        self._advanced_probing = False
        if self._advanced_probing == True:
            self._probing_quantities["gvec"]=["gvec",1]
            self._probing_quantities["gmag"]=["gmag",1]
            self._probing_quantities["nproj"]=["nproj",1]
            self._probing_quantities["nperp"]=["nperp",1]
            self._probing_quantities["lapl"]=["lapl",1]
            self._probing_quantities["hf"]=["hf",0]
            self._probing_quantities["2dd"]=["2dd",1]
            self._probing_quantities["geoten"]=["geoten",1]
            self._probing_quantities["kappa1"]=["kappa1",1]
            self._probing_quantities["kappa2"]=["kappa2",1]
            self._probing_quantities["totalcurv"]=["totalcurv",1]
            self._probing_quantities["st"]=["st",1]
            self._probing_quantities["si"]=["si",1]
            self._probing_quantities["meancurv"]=["meancurv",1]
            self._probing_quantities["gausscurv"]=["gausscurv",1]
            self._probing_quantities["curvdir1"]=["curvdir1",1]
            self._probing_quantities["curvdir2"]=["curvdir2",1]
            self._probing_quantities["flowlinecurv"]=["flowlinecurv",1]
            self._probing_quantities["median"]=["median",1]
            self._probing_quantities["hvalley"]=["hvalley",1]
            self._probing_quantities["hridge"]=["hridge",1]
            self._probing_quantities["hdpeak"]=["hdpeak",1]

    def set_vol_params(self):
        if self._single_scale == 0:
            self._volParams = " -vol \"{_sp_in_file_name}\":scalar:0-{_scale_samples}-{_max_scale}-o:V \"{_sp_in_file_name}\"" \
                              ":scalar:0-{_scale_samples}-{_max_scale}-on:VSN".format(
                _sp_in_file_name=self._sp_in_file_name,
                _scale_samples=self._scale_samples,
                _max_scale=self._max_scale
            )

        else:
            self._volParams = " -vol \"{}\":scalar:V".format(self._sp_in_file_name)

        if self._use_mask == True and self._sp_mask_file_name is not None:
            self._volParams += " \"{}\":scalar:M".format(self._sp_mask_file_name)

        if self._permissive == True:
            self._volParams += " -usa true"

    def set_info_params(self):
        if self._single_scale == 1:
            volTag = "V"
        else:
            volTag = "VSN"

        if self._feature_type == "ridge_line":
            self._info_params = (
            " -info h-c:V:val:0:-1 hgvec:V:gvec \
                hhess:V:hess tan1:V:hevec1 tan2:V:hevec2 ")
            self._info_params += (
                "sthr:" + volTag + ":heval1:" + str(self._seed_thresh) + ":-1 "
                "lthr:" + volTag + ":heval1:" + str(self._live_thresh) + ":-1 ")
            self._info_params += (
                 "strn:" + volTag + ":heval1:0:-1 ")
            if self._use_mode_thresh == True:
                self._info_params += (
                    "lthr2:" + volTag + ":hmode:" + \
                    str(self._mode_thresh) + ":1 ")
        elif self._feature_type == "valley_line":
            self._info_params = (
                " -info h-c:V:val:0:1  hgvec:V:gvec \
                hhess:V:hess tan1:V:hevec0 tan2:V:hevec1 ")
            self._info_params += (
                "sthr:" + volTag + ":heval1:" + str(self._seed_thresh) + ":1 "
                "lthr:" + volTag + ":heval1:"+str(self._live_thresh) + ":1 ")
            self._info_params += (
                "strn:" + volTag + ":heval1:0:1 ")
            if self._use_mode_thresh == True:
                self._info_params += (
                    "lthr2:" + volTag + ":hmode:" + str(self._mode_thresh) + ":-1 ")
        elif self._feature_type == "ridge_surface":
            self._info_params = (" -info h-c:V:val:0:-1 hgvec:V:gvec \
            hhess:V:hess tan1:V:hevec2 ")
            self._info_params += (" sthr:" + volTag + ":heval2:" + 
                str(self._seed_thresh) + ":-1 lthr:" + volTag + ":heval2:")
            self._info_params += (str(self._live_thresh) + ":-1 lthr2:" + 
                volTag + ":hmode:" + str(self._mode_thresh) + ":-1 strn:" + 
                volTag + ":heval2:0:-1 ")
            #Here we don't give an option for mode thresh. Always active
        elif self._feature_type == "valley_surface":
            self._info_params = (" -info h-c:V:val:0:1 hgvec:V:gvec \
            hhess:V:hess tan1:V:hevec0 ")
            self._info_params += (" sthr:" + volTag + ":heval0:" + 
                str(self._seed_thresh) + ":1 lthr:" + volTag + ":heval0:")
            self._info_params += (str(self._live_thresh) + ":1 lthr2:" + 
                volTag + ":hmode:" + str(self._mode_thresh) + 
                ":1 strn:" + volTag + ":heval0:0:1 ")
            #Here we don't give an option for mode thresh. Always active

        if self._use_mask == True:
            maskVal = "0.5"
            self._info_params += " spthr:M:val:" + maskVal + ":1"

    def set_kernel_params(self):
        if self._recon_kernel_type == "Bspline3":
            #self._reconKernelParams = "-k00 cubic:1,0 -k11 \
            #cubicd:1,0 -k22 cubicdd:1,0 -kssr hermite"
            self._reconKernelParams = \
                "-k00 bspl3 -k11 bspl3d -k22 bspl3dd"
            self._inverse_kernel_params = "-k bspl3ai"
        elif self._recon_kernel_type == "Bspline5":
            self._reconKernelParams = \
                "-k00 bspl5 -k11 bspl5d -k22 bspl5dd"
            self._inverse_kernel_params = "-k bspl5ai"
        elif self._recon_kernel_type == "Bspline7":
            self._reconKernelParams = \
                "-k00 bspl7 -k11 bspl7d -k22 bspl7dd"
            self._inverse_kernel_params = "-k bspl7ai"
        elif self._recon_kernel_type == "C4":
            self._reconKernelParams = \
                "-k00 c4h -k11 c4hd -k22 c4hdd"
            self._inverse_kernel_params = "-k c4hai"
        else:
            self._reconKernelParams = \
                "-k00 c4h -k11 c4hd -k22 c4hdd"
            self._inverse_kernel_params = "-k c4hai"
        
        # Add recon kernel along scale if we run multiscale mode
        if self._single_scale == 0:
            self._reconKernelParams = self._reconKernelParams + " -kssr hermite"
            if self._blurring_kernel_type == "DiscreteGaussian":
                self._reconKernelParams = self._reconKernelParams + \
                  " -kssb ds:1,5"
            elif self._blurring_kernel_type == "ContinuousGaussian":
                self._reconKernelParams = self._reconKernelParams + \
                  " -kssb gauss:1,5"

    def set_optimizer_params(self):
        self._optimizerParams = \
            "-pcp "+ str(self._population_control_period) + " -edpcmin 0.1 \
            -edmin 0.0000001 -eip 0.00001 -ess 0.5 -oss 2.0 -step 1 -maxci 10 \
            -rng 45 -bws "+ str(self._binning_width)
        
        if self._no_add == 1:
          self._optimizerParams += " -noadd"

    def set_energy_params(self):
        self._energyParams = \
            "-enr qwell:0.7 -ens bparab:10,0.7,-0.00 -enw butter:10,0.7"
        self._energyParams += " -efs " + str(self._use_strength).lower()
        self._energyParams += " -int "+ self._inter_particle_energy_type
        self._energyParams += " -irad "+ str(self._irad)
        self._energyParams += " -srad "+ str(self._srad)
        self._energyParams += " -alpha " + str(self._alpha)
        self._energyParams += " -beta " + str(self._beta)
        self._energyParams += " -gamma "+ str(self._gamma)

    def set_init_params(self):
        if self._init_mode == "Random":
            self._init_params = "-np "+ str(self._number_init_particles)
        elif self._init_mode == "Halton":
            self._init_params = "-np "+ str(self._number_init_particles) + "-halton"
        elif self._init_mode == "Particles":
            self._init_params = '-pi "%s"' % self._in_particles_file_name
        elif self._init_mode == "PerVoxel":
            self._init_params = " -ppv " + str(self._ppv) + " -nss " + \
                str(self._nss) + " -jit " + str(self._jit)

    def set_misc_params(self):
        if self._verbose == 0:
            verbose = 0
        else:
            verbose = self._verbose -1
        self._miscParams="-nave true -v "+str(verbose)+" -pbm 0"

        if self._scale_samples > 1:
            #Set boundary behaviour for scale-space blurring to bleed (gprobe 
            # uses that). This allows us to reuse the blurStack when probing
            self._miscParams = self._miscParams + " -bsp bleed"

    def reset_params(self):
        self._info_params = ""
        self._volParams  = ""
        self._reconKernelParams = ""
        self._inverse_kernel_params = ""
        self._optimizerParams = ""
        self._energyParams = ""
        self._init_params = ""
        self._miscParams = ""

    def build_params(self):
        self.set_info_params()
        self.set_vol_params()
        self.set_optimizer_params()
        self.set_energy_params()
        self.set_kernel_params()
        self.set_init_params()
        self.set_misc_params()

    def execute(self):
      
        self.preprocessing()
      
        self.build_params()
        outputParticles=os.path.join(self._tmp_dir, \
                                     self._tmp_particles_file_name)
        self.execute_pass(outputParticles)
        self.probe_quantities(self._sp_in_file_name, outputParticles)

        #Save NRRD data to VTK
        self.save_vtk(outputParticles)

        #Clean tmp Directory
        self.clean_tmp_dir()

    def execute_pass(self, output):
        #Check inputs files are in place
        if os.path.exists(self._in_file_name) == False:
            return False

        if self._use_mask==True and self._sp_mask_file_name is not None:
            if os.path.exists(self._sp_mask_file_name) == False:
                return False

        if self._single_scale == 1:
            tmp_command =  "unu resample -i \"{_in_file_name}\" -s x1 x1 x1 -k dgauss:{_max_scale},3 -t float -o \"{_sp_in_file_name}\"".format(
                _in_file_name=self._in_file_name, _max_scale=self._max_scale, _sp_in_file_name=self._sp_in_file_name)

            if self._debug == True:
                print (tmp_command)

            subprocess.call(tmp_command, shell=True)
            
        tmp_command = "puller -sscp \"{_tmp_dir}\" -cbst true {_volParams} {_miscParams} " \
                      "{_info_params} {_energyParams} " \
                      "{_init_params} {_reconKernelParams} " \
                      "{_optimizerParams} -o \"{output}\" -maxi {_iterations}".format(
            _tmp_dir=self._tmp_dir, _volParams=self._volParams, _miscParams=self._miscParams,
            _info_params=self._info_params, _energyParams=self._energyParams,
            _init_params=self._init_params,  _reconKernelParams=self._reconKernelParams,
            _optimizerParams=self._optimizerParams, output=output, _iterations=self._iterations)

        
        if self._debug == True:
            print (tmp_command)

        subprocess.call(tmp_command, shell=True)
        
        # Trick to add scale value
        if self._single_scale == 1:
            tmp_command = "unu crop -min M 0 -max M M -i \"{output}\"" \
                            " | unu 2op + - {_max_scale}" \
                            " | unu inset -min M 0 -s - -i \"{output}\" -o \"{output}\"".format(
                                output=output, _max_scale=self._max_scale)

        
            if self._debug == True:
                print (tmp_command)
            subprocess.call( tmp_command, shell=True )
          
    def probe_points(self, in_volume, inputParticles, quantity,
                     normalizedDerivatives=0):
        output = os.path.join(self._tmp_dir, quantity+".nrrd")
        if self._single_scale == 1:
            tmp_command = "unu crop -i \"{inputParticles}\" -min 0 0 -max 2 M | gprobe -i \"{in_volume}\" -k scalar " \
                          "{_reconKernelParams} -pi - -q {quantity} -v {verbose} -o \"{output}\"".format(
                inputParticles=inputParticles, in_volume=in_volume, _reconKernelParams=self._reconKernelParams,
                quantity=quantity, verbose=self._verbose, output=output)
        else:
            tmp_command = (
                "gprobe -i \"%(input)s\" "
                "-k scalar %(kernel_params)s -pi \"%(input_particles)s\" "
                "-q %(qty)s -v 0 -o \"%(output)s\" "
                "-ssn %(num_scales)d -sso -ssr 0 %(max_scale)03u "
                "-ssf \"%(path_name)s\"-%%03u-\"%(scale_samples)03u.nrrd\""
            ) % {
                'input': in_volume,
                'kernel_params': self._reconKernelParams,
                'input_particles': inputParticles,
                'qty': quantity,
                'output': output,
                'num_scales': self._scale_samples,
                'max_scale': self._max_scale,
                'scale_samples': self._scale_samples,
                'path_name': os.path.join(self._tmp_dir,"V")
            }

            if normalizedDerivatives == 1:
                tmp_command += " -ssnd"

        if self._debug == True:
          print (tmp_command)

        #print tmp_command
        subprocess.call( tmp_command, shell=True )

    def probe_quantities(self, in_volume, in_particles):
        """
        Parameters
        ----------
        in_volume : string
        
        in_particles : string
        """
        for quant in self._probing_quantities.keys():
            self.probe_points(in_volume, in_particles, quant, self._probing_quantities[quant][1])

    def preprocessing(self):
        if self._down_sample_rate > 1:
            downsampled_vol = os.path.join(self._tmp_dir, "ct-down.nrrd")
            self.down_sample(self._in_file_name,downsampled_vol,'cubic:0,0.5',self._down_sample_rate)
            if self._use_mask == True:
                downsampled_mask = os.path.join(self._tmp_dir, "mask-down.nrrd")
                self.down_sample(self._mask_file_name,downsampled_mask,'cheap',self._down_sample_rate)
                self._sp_mask_file_name = downsampled_mask
        else:
            downsampled_vol = self._in_file_name
            self._sp_mask_file_name = self._mask_file_name

        deconvolved_vol = os.path.join(self._tmp_dir, "ct-deconv.nrrd")
        self.deconvolve(downsampled_vol,deconvolved_vol)

        self._sp_in_file_name = deconvolved_vol
        
    def deconvolve(self, in_vol, out_vol):
        """
        Parameters
        ----------
        in_vol : string

        out_vol : string
        
        """
        tmp_command = "unu 3op clamp {min_intensity} \"{in_vol}\" {max_intensity} " \
                      " | unu resample -s x1 x1 x1 {inverse_kernel_params} -t float -o \"{out_vol}\"".format(
                              min_intensity=self._min_intensity,
                              in_vol=in_vol,
                              max_intensity=self._max_intensity,
                              inverse_kernel_params=self._inverse_kernel_params,
                              out_vol=out_vol)

        if self._debug == True:
            print (tmp_command)

        subprocess.call( tmp_command, shell=True)

    def down_sample(self, input_vol, output_vol, kernel, down_rate):
        #MAYBE WE HAVE TO DOWNSAMPLE THE MASK
        val = 1.0/down_rate
        # tmp_command = tmp_command %  {'rate':val,'kernel':kernel}

        tmp_command = "unu resample -s x{rate} x{rate} x{rate} -k {kernel} -i \"{input_vol}\" -o \"{output_vol}\"".format(
            rate=val, kernel=kernel, input_vol=input_vol, output_vol=output_vol)

        if self._debug == True:
            print (tmp_command)

        subprocess.call( tmp_command, shell=True)
    

    def adjust_scale(self, in_particles):
        #Trick to multiply scale if we have down-sampled before saving to VTK
        if self._down_sample_rate > 1:
            tmp_command = "unu crop -i \"%(output)s\" -min 3 0 -max 3 M | \
            unu 2op x - %(rate)f | unu inset -i \"%(output)s\" -s - \
            -min 3 0 -o \"%(output)s\"" % {'output': in_particles, 'rate': self._down_sample_rate}
        
            if self._debug == True:
                print (tmp_command)
            subprocess.call( tmp_command, shell=True )


    def save_vtk(self, in_particles,out_particles=None):
      
        if out_particles == None:
            out_particles = self._out_particles_file_name
            
        #Adjust scale if down-sampling was performed
        if self._down_sample_rate > 1:
            self.adjust_scale(in_particles)

        reader_writer = ReadNRRDsWriteVTK(out_particles)
        reader_writer.add_file_name_array_name_pair(in_particles, "NA")

        for quant in self._probing_quantities.keys():
            file = os.path.join(self._tmp_dir,"%s.nrrd" % quant)
            vtk_tag=self._probing_quantities[quant][0]
            reader_writer.add_file_name_array_name_pair(file,vtk_tag)
        
        #Recover spacing info from input volume
        image_io = ImageReaderWriter()
        im=image_io.read(self._sp_in_file_name)
        self._spacing = im.GetSpacing()
        del im
        
        #Include metadata information
        reader_writer.add_metadata_name_value_pair("irad", self._irad)
        reader_writer.add_metadata_name_value_pair("srad", self._srad)
        reader_writer.add_metadata_name_value_pair("liveth", self._live_thresh)
        reader_writer.add_metadata_name_value_pair("seedth", self._seed_thresh)
        reader_writer.add_metadata_name_value_pair("spacing", self._spacing)
        reader_writer.set_cip_region( self._cip_region )
        reader_writer.set_cip_type( self._cip_type )
        reader_writer.execute()

    def clean_tmp_dir(self):
        if self._clean_tmp_dir == True:
            print ("Cleaning temporary directory...")
            
#            tmp_command = "\bin\rm " + os.path.join(self._tmp_dir, "*")
#            print tmp_command
#
#            if self._debug == True:
#                print tmp_command
#            subprocess.call( tmp_command, shell=True )
            list_dir=os.listdir(self._tmp_dir)
            for file in list_dir:
                tmp = os.path.join(self._tmp_dir, file)
                os.remove(tmp)

    def merge_particles(self, input_list, output_merged):
        tmp_command = "unu join -a 1 -i \"{particles}\" -o \"{output_merged}\"".format(
                        particles=" ".join(map(str, input_list)),
                        output_merged=output_merged
        )

        if self._debug == True:
            print (tmp_command)
        subprocess.call(tmp_command, shell=True)

    def differential_mask (self, current_down_rate, previous_down_rate, output_mask):
        if self._use_mask == True:
            downsampled_mask_prev = os.path.join(self._tmp_dir, "mask-down-previous.nrrd")
            #Down-sampling previous level
            self.down_sample(self._mask_file_name, \
                     downsampled_mask_prev, "cheap", previous_down_rate)
            #Up-sampling previous level
            self.down_sample(downsampled_mask_prev, \
                     downsampled_mask_prev, "cheap", 1.0/previous_down_rate)
            #Down-sampling current level    
            self.down_sample(self._mask_file_name, \
                     output_mask, "cheap",current_down_rate)
            tmp_command = "unu 2op - \"%(current)s\" \"%(previous)s\" | unu 1op abs \
                           -i - | unu 2op gt - 0 -o \"%(out)s\"" % \
                          {'current': output_mask,
                            'previous': downsampled_mask_prev,
                            'out': output_mask}

            if self._debug == True:
                print (tmp_command)
            subprocess.call(tmp_command, shell=True)


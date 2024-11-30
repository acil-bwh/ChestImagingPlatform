#!/usr/bin/python

#------------------------------------------------------------------------------------
# TODO: We will want to put some of the functionality into a base class from which
# 'Generate[Airway,Vessel,Fissure]Particles' can inherit.
#
# TODO: Create a separate class for probing points (post-processing unu calls)
#------------------------------------------------------------------------------------

import subprocess
import os
from subprocess import PIPE

class FeatureStrengthMap:
    """Class to compute a feature strenght map based on the hessian eigenvalues
        
    Paramters
    ---------
    feature_type : string
    Takes on one or four values: "ridge_line" (for vessels), "valley_line"
    (for airways), "ridge_surface" (for fissure), "valley_surface"
        
    in_file_name : string
    File name of input volume
        
    out_particles_file_name : string
    File name of the output feature strenght map
        
    tmp_dir : string
    Name of temporary directory in which to store intermediate files

    """
    
    def __init__( self, feature_type,
                 in_file_name,out_file_name,
                 tmp_dir):
        
        assert feature_type == "ridge_line" or feature_type == "valley_line" \
            or feature_type == "ridge_surface" or feature_type == \
            "valley_surface", "Invalid feature type"

        self._feature_type = feature_type
        self._input_file_name = in_file_name
        self._output_file_name = out_file_name
        self._tmp_dir = tmp_dir
        self._reconKernelParams            = ""
        self._reconInverseKernelParams     = ""
                
        # If set to true, the temporary directory will be wiped clean
        # after execution
        self._clean_tmp_dir = False

        #
        # Particle system set up params
        #
        self._max_feature_strength = -1000           #Maximum value for the feature strength (need to know to do some clamping)
        self._reconKernelType    = "C4"        #Options: Bspline3, Bspline5, Bspline7 and C4
        self._max_scale           = 6           #TODO: Description
        self._scale_samples       = 5          #TODO: Description
        self._probe_scales                = [1,2,3]

        self._verbose                    = 1

        #
        #Pre proc Parameters
        #
        self._maxIntensity     =  400  #Limit the uppe range of the input image
        self._minIntensity     = -1024 #Limit the lowerc range of the input image
        self._downSamplingRate =  1    #Downsampling factor to enable multiresolution particles

    def set_kernel_params(self):
        if self._reconKernelType == "Bspline3":
            #self._reconKernelParams="-k00 cubic:1,0 -k11 cubicd:1,0 -k22 cubicdd:1,0 -kssr hermite"
            self._reconKernelParams= "-k00 bspl3 -k11 bspl3d -k22 bspl3dd -kssr hermite"
            self._reconInverseKernelParams = "-k bspl3ai"
        elif self._reconKernelType == "Bspline5":
            self._reconKernelParams="-k00 bspl5 -k11 bspl5d -k22 bspl5dd -kssr hermite"
            self._reconInverseKernelParams = "-k bspl5ai"
        elif self._reconKernelType == "Bspline7":
            self._reconKernelParams="-k00 bspl7 -k11 bspl7d -k22 bspl7dd -kssr hermite"
            self._reconInverseKernelParams = "-k bspl7ai"
        elif self._reconKernelType == "C4":
            self._reconKernelParams="-k00 c4h -k11 c4hd -k22 c4hdd -kssr hermite"
            self._reconInverseKernelParams = "-k c4hai"
        else:
            self._reconKernelParams="-k00 c4h -k11 c4hd -k22 c4hdd -kssr hermite"
            self._reconInverseKernelParams = "-k c4hai"

    def get_feature_strength_from_feature_type (self, feature_type):
        if feature_type == "ridge_line":
            return "heval1"
        elif feature_type == "valley_line":
            return "heval1"
        elif feature_type == "ridge_surface":
            return "heval2"
        elif feature_type == "valley_surface":
            return "heval0"

    def execute( self ):
        if self._downSamplingRate > 1:
            downsampledVolume = os.path.join(self._tmp_dir,"ct-down.nrrd")
            self.down_sample(self._input_file_name,downsampledVolume)
        else:
            downsampledVolume = self._input_file_name

        deconvolvedVolume = os.path.join(self._tmp_dir,"ct-deconv.nrrd")
        self.deconvolution(downsampledVolume,deconvolvedVolume)

        inputVolume = deconvolvedVolume

        #Probe scales
        responseFiles = ""
        for ii in range(len(self._probe_scales)):
            outputVolume = "kk-%d.nhdr" % ii
            outputVolume = os.path.join(self._tmp_dir,outputVolume)
            pS = self._probe_scales[ii]
            self.probe_volume(inputVolume,outputVolume,pS,1)
            responseFiles += outputVolume
            responseFiles += " "

        #Join scale responses and compute maximum strength
        if self._feature_type == "ridge_line":
            operator = "min"
            clampRange = "%f - 0"
        elif self._feature_type == "valley_line":
            operator = "max"
            clampRange = "0 - %f"
        elif self._feature_type == "ridge_surface":
            operator = "min"
            clampRange = "%f - 0"
        elif self._feature_type == "valley_surface":
            operator = "max"
            clampRange = "0 - %f"

        clampRange = clampRange % self._max_feature_strength

        tmpCommand = "unu join -i %(input)s -incr -a 0 -sp 1 | unu 3op clamp %(clamp_range)s | unu project -m %(operator)s -a 0 -o %(output)s"
        tmpCommand = tmpCommand % {'input':responseFiles,'output':self._output_file_name,'operator':operator,'clamp_range':clampRange}
        subprocess.call( tmpCommand, shell=True )

        #Trick to fix space directions: project wipes out this information
        tmpCommand = "unu 2op gt %(input)s -3000 | unu 2op x - %(output)s -w 0 -o %(output)s -t float"
        tmpCommand = tmpCommand % {'input':self._input_file_name,'output':self._output_file_name}
        subprocess.call( tmpCommand, shell=True )

        self.clean_tmp_dir()

    def probe_volume( self, inputVolume,outputVolume, probe_scale, normalizedDerivatives=0 ):

        featureStrength= self.get_feature_strength_from_feature_type(self._feature_type)

            #tmpCommand = (
            #    "cd "+ self._tmp_dir + "; gprobe -i " + inputVolume +
            #    "-k scalar " + self._reconKernelParams + "-pi " + inputParticles +
            #    "-q " + quantity + "-v 0 -o " + output +
            #    "-sso -ssr 0 %03u" % self._maxScale + "-ssf V-%03u-"
            #    + "%03u" % self._scaleSamples +".nrrd"
            #)
        tmpCommand = (
                        "cd %(temporary_directory)s; vprobe -i %(input)s "
                        "-k scalar %(kernel_params)s "
                        "-q %(qty)s -v 0 -o %(output)s "
                        "-ssn %(num_scales)d -sso -ssr 0 %(max_scale)03u "
                        "-sssf V-%%03u-%(scale_samples)03u.nrrd -ssw %(probe_scale)f"
                                ) % {
                        'temporary_directory': self._tmp_dir,
                        'input': inputVolume,
                        'kernel_params': self._reconKernelParams,
                        'qty': featureStrength,
                        'output': outputVolume,
                        'num_scales': self._scale_samples,
                        'max_scale': self._max_scale,
                        'scale_samples': self._scale_samples,
                        'probe_scale': probe_scale
                                }
        if normalizedDerivatives == 1:
            tmpCommand += " -ssnd"

        if self._verbose == 1:
            print (tmpCommand)

        subprocess.call( tmpCommand, shell=True )


    def deconvolution(self,inputVol,outputVol):
        tmpCommand = "unu 3op clamp " + str(self._minIntensity) + " " + inputVol + " " + str(self._maxIntensity)  + \
                        " | unu resample -s x1 x1 x1 " + self._reconInverseKernelParams + " -t float -o " + outputVol
        if self._verbose == 1:
            print (tmpCommand)
        subprocess.call( tmpCommand, shell=True)

    def down_sample(self,inputVol,outputVol):
        tmpCommand = "unu resample -s x%(rate)f x%(rate)f x%(rate)f -k cubic:0,0.5 -i "+ inputVol + " -o " + outputVol
        #MAYBE WE HAVE TO DOWNSAMPLE THE MASK
        val=1.0/self._downSamplingRate
        tmpCommand = tmpCommand %  {'rate':val}

        if self._verbose == 1:
            print (tmpCommand)

        subprocess.call( tmpCommand, shell=True)

    def clean_tmp_dir(self):
        if self._clean_tmp_dir == True:
            print ("Cleaning temporary directory...")
            tmpCommand = "/bin/rm " + os.path.join(self._tmp_dir,"*")
            subprocess.call( tmpCommand, shell=True )

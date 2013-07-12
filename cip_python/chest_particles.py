#!/usr/bin/python

#------------------------------------------------------------------------------------
# TODO: We will want to put some of the functionality into a base class from which
# 'Generate[Airway,Vessel,Fissure]Particles' can inherit.
#
# TODO: Create a separate class for probing points (post-processing unu calls)
#------------------------------------------------------------------------------------

import pdb
import subprocess
import os
from subprocess import PIPE
from cip_python.ReadNRRDsWriteVTK import ReadNRRDsWriteVTK

class GenerateParticles:
    """Base class for airway, vessel, and fissure particles classes.

    Parameters
    ----------
    feature_type : string
        Takes on one or four values: "RidgeLine" (for vessels), "ValleyLine"
        (for airways), "RidgeSurface" (for fissure), "ValleySurface"
    """
    def __init__(self, feature_type):
        assert feature_type == "RidgeLine" or feature_type == "ValleyLine" or \
            feature_type == "RidgeSurface" or feature_type == "ValleySurface",
        "Invalid feature type"
        
        self._feature_type = feature_type        
        self._reconInverseKernelParams     = ""
        self._initParams                   = ""
        self.tmp_dir                       = "."
        self.in_file_name                = "NA"
        self._maskFileName                 = "NA"
        self._inputNRRDParticlesFileName   = "NA"
        self._particlesFileName            = "NA"

        # If set to true, the temporary directory will be wiped clean
        # after execution
        self.clean_tmp_dir = False

        # Temporal filenames
        self._inputVolume = self.in_file_name  #Volume name that is going to be used for scale-space particles. This volume is the result of any pre-processing
        self._nrrdParticlesFileName    = "nrrdParticlesFileName.nrrd"  #Nrrd file to contain particle data.

        # Particle system set up param. 
        self._reconKernelType    = "C4"        #Options: Bspline3, Bspline5, Bspline7 and C4
        self._initializationMode = "Random"    #Options: Ramdom, Halton, Particles, PerVoxel
        self._singleScale        = 0           #Run at a single scale vs a full scale-space exploration
        self._useMask            = 1
        self._maxScale           = 6           #TODO: Description
        self._scaleSamples       = 5          #TODO: Description
        self._iterations         = 50

        self._verbose                    = 1

        # Energy specific params
        self._irad  = 1.7
        self._srad  = 1.2
        self._alpha = 0.5
        self._beta  = 0.5
        self._gamma = 1
        self._interParticleEnergyType = "justr" #Options: justr, add
        self._useStrength = "true"

        # Init specific params
        self._ppv = 1 # Points per voxel.
        self._nss = 1 # Number of samples along scale axis
        self._jit = 1 # Jittering to do for each point

        # Optimizer params
        self._binningWidth = 1.2 #Binning width as mulitple of irad. Increase this value run into overflow of maximum number of bins.
        self._populationControlPeriod = 5 #Population control period
        self._noAdd = 0 #If enabled, it does not add points during population control

        # Pre proc Parameters
        self._maxIntensity     =  400  #Limit the uppe range of the input image
        self._minIntensity     = -1024 #Limit the lowerc range of the input image
        self._downSamplingRate =  1    #Downsampling factor to enable multiresolution particles

        # Basic contrast Parameters
        self._liveThreshold  = -150 #Threshold on feature strength
        self._seedThreshold  = -100 #Threshold on feature strengh
        self._modeThreshold  = -0.5 #Threshold on mode of the hessian

    def SetFeatureTypeToValleySurface( self ):
        self._featureType = "ValleySurface"

    def SetFeatureTypeToRidgeSurface( self ):
        self._featureType = "RidgeSurface"

    def SetFeatureTypeToValleyLine( self ):
        self._featureType = "ValleyLine"

    def SetFeatureTypeToRidgeLine( self ):
        self._featureType = "RidgeLine"

    def SetInputNRRDParticles ( self, inputNRRDParticlesFileName):
        self._inputNRRDParticlesFileName = inputNRRDParticlesFileName

    def SetMaskFileName( self, maskFileName ):
        self._maskFileName = maskFileName

    def SetOutputFileName( self, particlesFileName ):
        self._particlesFileName = particlesFileName

    def SetInterParticleDistance( self, distance ):
        self._interParticleDistance = distance

    def SetVolParamGroup( self ):	   
        if self._singleScale == 0:
            self._volParams = " -vol " + self._inputVolume + ":scalar:0-" + str(self._scaleSamples) + "-" + str(self._maxScale) + "-o:V " \
            +  self._inputVolume + ":scalar:0-" + str(self._scaleSamples) + "-" + str(self._maxScale) + "-on:VSN"
        else:
            self._volParams = " -vol " +  self._inputVolume + ":scalar:V -usa true"

        if self._useMask == 1:
	   self._volParams += " " + self._maskFileName + ":scalar:M"

    def SetInfoParamGrop( self ):
        if self._singleScale == 1:
            volTag = "V"
        else:
            volTag = "VSN"

        if self._featureType == "RidgeLine":
            self._infoParams = " -info  h-c:V:val:0:-1  hgvec:V:gvec  hhess:V:hess tan1:V:hevec1 tan2:V:hevec2 "
            self._infoParams += "sthr:" + volTag + ":heval1:" + str(self._seedThreshold) + ":-1 lthr:" + volTag + ":heval1:"
            self._infoParams += str(self._liveThreshold) + ":-1 strn:" + volTag + ":heval1:0:-1"
        elif self._featureType == "ValleyLine":
            self._infoParams = " -info  h-c:V:val:0:1  hgvec:V:gvec  hhess:V:hess tan1:V:hevec0 tan2:V:hevec1 "
            self._infoParams += "sthr:" + volTag + ":heval1:" + str(self._seedThreshold) + ":1 lthr:" + volTag + ":heval1:"
            self._infoParams += str(self._liveThreshold) + ":1 strn:" + volTag + ":heval1:0:1"
        elif self._featureType == "RidgeSurface":
            self._infoParams = " -info h-c:V:val:0:-1  hgvec:V:gvec  hhess:V:hess tan1:V:hevec2 "
            self._infoParams += " sthr:" + volTag + ":heval2:" + str(self._seedThreshold) + ":-1 lthr:" + volTag + ":heval2:"
            self._infoParams += str(self._liveThreshold) + ":-1 lthr2:" + volTag + ":hmode:" + str(self._modeThreshold) + ":-1 strn:" + volTag + ":heval2:0:-1"
        elif self._featureType == "ValleySurface":
            self._infoParams = " -info h-c:V:val:0:1  hgvec:V:gvec  hhess:V:hess tan1:V:hevec0 "
            self._infoParams +=" sthr:" + volTag + ":heval0:" + str(self._seedThreshold) + ":1 lthr:" + volTag + ":heval0:"
            self._infoParams += str(self._liveThreshold) + ":1 lthr2:" + volTag + ":hmode:" + str(self._modeThreshold) + ":1 strn:" + volTag + ":heval0:0:1"

        if self._useMask == 1:
	    maskVal = "0.5";
	    self._infoParams += " spthr:M:val:" + maskVal + ":1"

    def SetReconKernelParamsGroup(self):
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

    def SetOptimizerParamsGroup( self ):
        self._optimizerParams = "-pcp "+ str(self._populationControlPeriod) + " -edpcmin 0.1 -edmin 0.0000001 -eip 0.001 -ess 0.2 -oss 1.9 -step 1 -maxci 10 -rng 45 -bws "+ str(self._binningWidth)
        if  self._noAdd == 1:
          self._optimizerParams += " -noadd"

    def SetEnergyParamsGroup(self):
        self._energyParams = "-enr qwell:0.7 -ens bparab:10,0.7,-0.00 -enw butter:10,0.7"
        self._energyParams += " -efs " + str(self._useStrength)
        self._energyParams += " -int "+ self._interParticleEnergyType
        self._energyParams += " -irad "+ str(self._irad)
        self._energyParams += " -srad "+ str(self._srad)
        self._energyParams += " -alpha " + str(self._alpha)
        self._energyParams += " -beta " + str(self._beta)
        self._energyParams += " -gamma "+ str(self._gamma)

    def SetInitParamsGroup(self):
        if self._initializationMode == "Random":
            self._initParams = "-np "+ str(self._numberParticles)
        elif self._initializationMode == "Halton":
            self._initParams = "-np "+ str(self._numberParticles) + "-halton"
        elif self._initializationMode == "Particles":
            self._initParams = "-pi "+ self._inputNRRDParticlesFileName
        elif self._initializationMode == "PerVoxel":
            self._initParams = " -ppv " + str(self._ppv) + " -nss " + str(self._nss) + " -jit " + str(self._jit)

    def SetMiscParamsGroup(self):
        self._miscParams="-nave true -v "+str(self._verbose)+" -pbm 0"

    def ResetParamGroups(self):
        self._infoParams = ""
        self._volParams  = ""
        self._reconKernelParams = ""
        self._reconInverseKernelParams = ""
        self._optimizerParams = ""
        self._energyParams = ""
        self._initParams = ""
        self._miscParams = ""

    def BuildParamGroups(self):
        self.SetInfoParamGrop()
        self.SetVolParamGroup()
        self.SetOptimizerParamsGroup()
        self.SetEnergyParamsGroup()
        self.SetReconKernelParamsGroup()
        self.SetInitParamsGroup()
        self.SetMiscParamsGroup()

    def Execute(self):
        if self._downSamplingRate > 1:
            downsampledVolume = os.path.join(self.tmp_dir, "ct-down.nrrd")
            self.DownSampling(self.in_file_name,downsampledVolume,'cubic:0,0.5')
            if self._useMask == 1:
                downsampledMask = os.path.join(self.tmp_dir, "mask-down.nrrd")
                self.DownSampling(self._maskFileName,downsampledMask,'cheap')
                self._maskFileName = downsampledMask
        else:
            downsampledVolume = self.in_file_name

        deconvolvedVolume = os.path.join(self.tmp_dir, "ct-deconv.nrrd")
        self.deconvolve(downsampledVolume,deconvolvedVolume)

        self._inputVolume = deconvolvedVolume
        self.BuildParamGroups()
        outputParticles=os.path.join(self.tmp_dir, self._nrrdParticlesFileName)
        self.ExecutePass(outputParticles)
        self.ProbeAllQuantities(deconvolvedVolume,outputParticles)
        self.SaveParticlesToVTK(outputParticles)

    def ExecutePass(self, output):
        #Check inputs files are in place
        if os.path.exists(self._inputVolume) == False:
            return False

        if self._useMask==1:
            if os.path.exists(self._maskFileName) == False:
                return False

        if self._singleScale == 1:
            tmpCommand = "unu resample -i " + self._inputVolume + \
                " -s x1 x1 x1 -k dgauss:" + str(self._maxScale) + \
                ",3 -t float -o " + self._inputVolume

            subprocess( tmpCommand, shell=True)

        tmpCommand = "puller -sscp " + self.tmp_dir + \
            " -cbst true " + self._volParams + " " + self._miscParams + " " + \
            self._infoParams + " " +  self._energyParams + " " + \
            self._initParams + " " + self._reconKernelParams + " " + \
            self._optimizerParams + " -o " + output + " -maxi " + \
            str(self._iterations)

        subprocess.call( tmpCommand, shell=True )

        # Trick to add scale value
        if self._singleScale == 1:
            tmpCommand = "unu head " + output + \
                " | grep size | awk '{split($0,a,\" \"); print a[3]}'"

            tmpNP = subprocess.Popen(tmpCommand, shell=True, stdout=PIPE,
                                      stderr=PIPE)
            NP = tmpNP.communicate()[0].rstrip('\n')
            tmpCommand = "echo \"0 0 0 " + str(self._maxScale) + \
                "\" | unu pad -min 0 0 -max 3 " + NP + \
                " -b mirror | unu 2op + - " + output + " -o " + output

            subprocess.call( tmpCommand, shell=True )

    def probe_points(self, inputVolume, inputParticles, quantity,
                     normalizedDerivatives=0):
        output = os.path.join(self.tmp_dir, quantity+".nrrd")
        if self._singleScale == 1:
            tmpCommand = "unu crop -i " + inputParticles + \
                "-min 0 0 -max 2 M | gprobe -i " + inputVolume + "-k scalar "
            tmpCommand += self._reconKernelParams + "-pi - -q " + \
                quantity + " -v 0 -o " + output
        else:
            #tmpCommand = (
            #    "cd "+ self._temporaryDirectory + "; gprobe -i " + inputVolume +
            #    "-k scalar " + self._reconKernelParams + "-pi " + inputParticles +
            #    "-q " + quantity + "-v 0 -o " + output +
            #    "-sso -ssr 0 %03u" % self._maxScale + "-ssf V-%03u-"
            #    + "%03u" % self._scaleSamples +".nrrd"
            #)
            tmpCommand = (
                "cd %(temporary_directory)s; gprobe -i %(input)s "
                "-k scalar %(kernel_params)s -pi %(input_particles)s "
                "-q %(qty)s -v 0 -o %(output)s "
                "-ssn %(num_scales)d -sso -ssr 0 %(max_scale)03u "
                "-ssf V-%%03u-%(scale_samples)03u.nrrd"
            ) % {
                'temporary_directory': self.tmp_dir,
                'input': inputVolume,
                'kernel_params': self._reconKernelParams,
                'input_particles': inputParticles,
                'qty': quantity,
                'output': output,
                'num_scales': self._scaleSamples,
                'max_scale': self._maxScale,
                'scale_samples': self._scaleSamples
            }

            if normalizedDerivatives == 1:
                tmpCommand += " -ssnd"
        if self._verbose == 1:
            print tmpCommand

        subprocess.call( tmpCommand, shell=True )

    def ProbeAllQuantities(self,inputVolume,inputParticles):
        self.probe_points(inputVolume, inputParticles, "val" )
        self.probe_points(inputVolume, inputParticles, "heval0", 1)
        self.probe_points(inputVolume, inputParticles, "heval1", 1)
        self.probe_points(inputVolume, inputParticles, "heval2", 1)
        self.probe_points(inputVolume, inputParticles, "hmode", 1)
        self.probe_points(inputVolume, inputParticles, "hevec0")
        self.probe_points(inputVolume, inputParticles, "hevec1")
        self.probe_points(inputVolume, inputParticles, "hevec2")
        self.probe_points(inputVolume, inputParticles, "hess")

    def deconvolve(self, inputVol, outputVol):
        """
        Parameters
        ----------
        in_vol : string

        out_vol : string
        
        """
        tmpCommand = "unu 3op clamp " + str(self._minIntensity) + " " + \
            in_vol + " " + str(self._maxIntensity)  + \
            " | unu resample -s x1 x1 x1 " + self._reconInverseKernelParams + \
            " -t float -o " + out_vol

        if self._verbose == 1:
            print tmpCommand

        subprocess.call( tmpCommand, shell=True)

    def DownSampling(self, inputVol, outputVol, kernel):    		
        tmpCommand = "unu resample -s x%(rate)f x%(rate)f x%(rate)f -k %(kernel)s -i "+ inputVol + " -o " + outputVol

        #MAYBE WE HAVE TO DOWNSAMPLE THE MASK
        val = 1.0/self._downSamplingRate
        tmpCommand = tmpCommand %  {'rate':val,'kernel':kernel}

        if self._verbose == 1:
            print tmpCommand

        subprocess.call( tmpCommand, shell=True)

    def SaveParticlesToVTK(self, inputParticles):
         #Trick to multiply scale if we have down-sampled before saving to VTK
        if self._downSamplingRate > 1:
            tmpCommand = "unu crop -i %(output)s -min 3 0 -max 3 M | unu 2op x - %(rate)f | unu inset -i %(output)s -s - -min 3 0 -o %(output)s"
            tmpCommand = tmpCommand % {'output':inputParticles, 'rate':self._downSamplingRate}
            print tmpCommand
            subprocess.call( tmpCommand, shell=True )

        readerWriter = ReadNRRDsWriteVTK()
        readerWriter.AddFileNameArrayNamePair( inputParticles,  "NA" )
        quantities=["val","heval0","heval1","heval2","hmode","hevec0","hevec1","hevec2","hess"]

        ##VTK field names should be standardized to match teem tags
        tags = ["val", "h0", "h1", "h2", "hmode", "hevec0", \
                "hevec1", "hevec2", "hess"]

        for ii in range(len(quantities)):
            file = os.path.join(self.tmp_dir,"%s.nrrd" % quantities[ii])
            readerWriter.AddFileNameArrayNamePair( file, tags[ii] )

        readerWriter.SetOutputFileName( self._particlesFileName )
        readerWriter.Execute()

    def CleanTemporaryDirecotry(self):
        if self._cleanTemporaryDirectory == True:
            print "Cleaning tempoarary directory..."
            tmpCommand = "rm " + os.path.join(self.tmp_dir, "*")
            subprocess.call( tmpCommand, shell=True )
            

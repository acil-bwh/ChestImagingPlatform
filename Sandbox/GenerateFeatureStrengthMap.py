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

class GenerateFeatureStrengthMap:
    def __init__( self ):
        self._reconKernelParams            = ""
        self._reconInverseKernelParams     = ""
        self._temporaryDirectory           = "."
        self._cipBuildDirectory            = ""
        self._cleanTemporaryDirectory      = True
        self._inputFileName                = "NA"
        self._outputFileName                       = "NA"

        #
        # Particle system set up params
        #
        self._featureType        = "RidgeLine" #This is a RidgeLine (Vessel), ValleyLine (Airway), RidgeSurface (Fissure), ValleySurface
        self._maxFeatureStrength = -1000           #Maximum value for the feature strength (need to know to do some clamping)
        self._reconKernelType    = "C4"        #Options: Bspline3, Bspline5, Bspline7 and C4
        self._maxScale           = 6           #TODO: Description
        self._scaleSamples       = 5          #TODO: Description
        self._probeScales                = [1,2,3]

        self._verbose                    = 1

        #
        #Pre proc Parameters
        #
        self._maxIntensity     =  400  #Limit the uppe range of the input image
        self._minIntensity     = -1024 #Limit the lowerc range of the input image
        self._downSamplingRate =  1    #Downsampling factor to enable multiresolution particles


    def SetFeatureTypeToValleySurface( self ):
        self._featureType = "ValleySurface"

    def SetFeatureTypeToRidgeSurface( self ):
        self._featureType = "RidgeSurface"

    def SetFeatureTypeToValleyLine( self ):
        self._featureType = "ValleyLine"

    def SetFeatureTypeToRidgeLine( self ):
        self._featureType = "RidgeLine"

    def SetCIPBuildDirectory( self, dir ):
        self._cipBuildDirectory = dir

    def SetInputFileName( self, inputFileName ):
        self._inputFileName = inputFileName

    def SetOutputFileName( self, outputFileName ):
        self._outputFileName = outputFileName

    def SetTemporaryDirectory( self, tempDir ):
        self._temporaryDirectory = tempDir

    def SetCleanTemporaryDirectory( self, tmp ):
        self._cleanTemporaryDirectory = tmp

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

    def GetFeatureStrengthFromFeatureType (self, featureType):
        if featureType == "RidgeLine":
            return "heval1"
        elif featureType == "ValleyLine":
            return "heval1"
        elif featureType == "RidgeSurface":
            return "heval2"
        elif featureType == "ValleySurface":
            return "heval0"

    def Execute( self ):
        if self._downSamplingRate > 1:
            downsampledVolume = os.path.join(self._temporaryDirectory,"ct-down.nrrd")
            self.DownSampling(self._inputFileName,downsampledVolume)
        else:
            downsampledVolume = self._inputFileName

        deconvolvedVolume = os.path.join(self._temporaryDirectory,"ct-deconv.nrrd")
        self.Deconvolution(downsampledVolume,deconvolvedVolume)

        inputVolume = deconvolvedVolume

        #Probe scales
        responseFiles = ""
        for ii in range(len(self._probeScales)):
            outputVolume = "kk-%d.nhdr" % ii
            outputVolume = os.path.join(self._temporaryDirectory,outputVolume)
            pS = self._probeScales[ii]
            self.ProbeVolume(inputVolume,outputVolume,pS,1)
            responseFiles += outputVolume
            responseFiles += " "

        #Join scale responses and compute maximum strength
        if self._featureType == "RidgeLine":
            operator = "min"
            clampRange = "%f - 0"
        elif self._featureType == "ValleyLine":
            operator = "max"
            clampRange = "0 - %f"
        elif self._featureType == "RidgeSurface":
            operator = "min"
            clampRange = "%f - 0"
        elif self._featureType == "ValleySurface":
            operator = "max"
            clampRange = "0 - %f"

        clampRange = clampRange % self._maxFeatureStrength

        tmpCommand = "unu join -i %(input)s -incr -a 0 -sp 1 | unu 3op clamp %(clamp_range)s | unu project -m %(operator)s -a 0 -o %(output)s"
        tmpCommand = tmpCommand % {'input':responseFiles,'output':self._outputFileName,'operator':operator,'clamp_range':clampRange}
        subprocess.call( tmpCommand, shell=True )

        #Trick to fix space directions: project wipes out this information
        tmpCommand = "unu 2op gt %(input)s -3000 | unu 2op x - %(output)s -w 0 -o %(output)s -t float"
        tmpCommand = tmpCommand % {'input':self._inputFileName,'output':self._outputFileName}
        subprocess.call( tmpCommand, shell=True )


    def ProbeVolume( self, inputVolume,outputVolume, probeScale, normalizedDerivatives=0 ):

        featureStrength= self.GetFeatureStrengthFromFeatureType(self._featureType)

            #tmpCommand = (
            #    "cd "+ self._temporaryDirectory + "; gprobe -i " + inputVolume +
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
                        'temporary_directory': self._temporaryDirectory,
                        'input': inputVolume,
                        'kernel_params': self._reconKernelParams,
                        'qty': featureStrength,
                        'output': outputVolume,
                        'num_scales': self._scaleSamples,
                        'max_scale': self._maxScale,
                        'scale_samples': self._scaleSamples,
                        'probe_scale': probeScale
                                }
        if normalizedDerivatives == 1:
            tmpCommand += " -ssnd"

        if self._verbose == 1:
            print tmpCommand

        subprocess.call( tmpCommand, shell=True )


    def Deconvolution(self,inputVol,outputVol):
        tmpCommand = "unu 3op clamp " + str(self._minIntensity) + " " + inputVol + " " + str(self._maxIntensity)  + \
                        " | unu resample -s x1 x1 x1 " + self._reconInverseKernelParams + " -t float -o " + outputVol
        if self._verbose == 1:
            print tmpCommand
        subprocess.call( tmpCommand, shell=True)

    def DownSampling(self,inputVol,outputVol):
        tmpCommand = "unu resample -s x%(rate)f x%(rate)f x%(rate)f -k cubic:0,0.5 -i "+ inputVol + " -o " + outputVol
        #MAYBE WE HAVE TO DOWNSAMPLE THE MASK
        val=1.0/self._downSamplingRate
        tmpCommand = tmpCommand %  {'rate':val}

        if self._verbose == 1:
            print tmpCommand

        subprocess.call( tmpCommand, shell=True)

    def CleanTemporaryDirecotry(self):
        if self._cleanTemporaryDirectory == True:
            print "Cleaning tempoarary directory..."
            tmpCommand = "rm " + os.path.join(self._temporaryDirectory,"*")
            subprocess.call( tmpCommand, shell=True )

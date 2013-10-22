#!/usr/bin/python

import subprocess
from subprocess import PIPE
from ReadNRRDsWriteVTK import ReadNRRDsWriteVTK

class GenerateFissureParticles:
        def __init__( self ):
                self._lowerThreshold          = -30.6
                self._mode                    = -0.5
                self._scale                   = 1.2
                self._numberParticles         = 10000
                self._numberIterations        = 50       
                self._interParticleDistance   = 4 
                self._seedThreshold           = -30.6
                self._cipBuildDirectory       = ""
                self._cleanTemporaryDirectory = True

        def SetCIPBuildDirectory( self, dir ):
                self._cipBuildDirectory = dir

        def SetInput( self, inputFileName ):
                self._inputFileName = inputFileName

        def SetOutputFileName( self, particlesFileName ):
                self._particlesFileName = particlesFileName

        def SetTemporaryDirectory( self, tempDir ):
                self._temporaryDirectory = tempDir

        def SetMask( self, mask ):
                self._mask = mask

        def SetInterParticleDistance( self, distance ):
                self._interParticleDistance = distance

        def SetCleanTemporaryDirectory( self, tmp ):
                self._cleanTemporaryDirectory = tmp

        def SetSeedThreshold( self, threshold ):
                self._seedThreshold = threshold

        def ProbePoints( self, quantity ):
                tmpCommand = "unu crop -i " + self._temporaryDirectory + "pass1.nhdr -min 0 0 -max 2 M | gprobe -i " + self._temporaryDirectory + "ct-" + str(self._scale) + ".nrrd -k scalar -k00 cubic:1,0 \
                        -k11 cubicd:1,0 -k22 cubicdd:1,0 -pi - -q " + quantity + " -v 0 -o " + self._temporaryDirectory + quantity + ".nrrd"
                subprocess.call( tmpCommand, shell=True )

        def Execute( self ):
                # TODO: Comment on the following operation and why it's needed
                print "Preprocessing mask..."
                tmpCommand = "unu 2op gt " + self._mask + " 0.5 -o " + self._temporaryDirectory + "lungMask.nrrd"
                subprocess.call( tmpCommand, shell=True )

                # TODO: Comment on the following operation and why it's needed
                print "Blurring volume..."
                tmpCommand = "unu resample -i " + self._inputFileName + " -s x1 x1 x1 -k dgauss:" + str(self._scale) + ",3 -t float -o " + self._temporaryDirectory + "ct-" + str(self._scale) + ".nrrd"
                subprocess.call( tmpCommand, shell=True )

                # TODO: Comment on these settings. Currently very cryptic
                volParams          = " -vol " + self._temporaryDirectory + "ct-" + str(self._scale) + ".nrrd:scalar:V " + self._temporaryDirectory + "lungMask.nrrd:scalar:M -usa true"
                reconKernelParams  = " -nave true -pbm 0 -k00 cubic:1,0 -k11 cubicd:1,0 -k22 cubicdd:1,0"
                optimizerParams    = " -pcp 5 -edpcmin 0.1 -edmin 0.0000001 -eip 0.001 -ess 0.2 -nss 5 -oss 1.9 -step 1 -maxci 10 -rng 43"
                energyParams       = " -efs false -lti true -nave true -cbst true -enr qwell:0.64 -int justr -irad " + str(self._interParticleDistance) + " -alpha 0.5 -beta 0"
                miscParams         = " -v 0"
                infoParams         = " -info h-c:V:val:0:-1  hgvec:V:gvec  hhess:V:hess tan1:V:hevec2 sthr:V:heval2:" + str(self._seedThreshold) + \
                        ":-1 spthr:M:val:0.5:1 lthr:V:heval2:" + str(self._lowerThreshold) + ":-1 lthr2:M:val:0.5:1 lthr3:V:hmode:" + str(self._mode) + ":-1 strn:V:heval2:0:-1"

                print "Running particles..."
                tmpCommand = "puller " + volParams + infoParams + energyParams + reconKernelParams + optimizerParams + miscParams + " -o " + self._temporaryDirectory + "pass1.nhdr -maxi " + \
                        str(self._numberIterations) + " -np " + str(self._numberParticles)
                subprocess.call( tmpCommand, shell=True )

                tmpCommand = "unu head " + self._temporaryDirectory + "pass1.nhdr | grep size | awk '{split($0,a,\" \"); print a[3]}'"
                tmpNP = subprocess.Popen( tmpCommand, shell=True, stdout=PIPE, stderr=PIPE )
                NP = tmpNP.communicate()[0].rstrip('\n')

                tmpCommand = "echo \"0 0 0 " + str(self._scale) + "\" | unu pad -min 0 0 -max 3 " + NP + " -b mirror | unu 2op + - " + self._temporaryDirectory + "pass1.nhdr -o " + self._temporaryDirectory + "pass1.nhdr"
                subprocess.call( tmpCommand, shell=True )
        
                print "Probing points..."
                self.ProbePoints( "val" )
                self.ProbePoints( "heval0" )
                self.ProbePoints( "heval1" )
                self.ProbePoints( "heval2" )
                self.ProbePoints( "hmode" )
                self.ProbePoints( "hevec0" )
                self.ProbePoints( "hevec1" )
                self.ProbePoints( "hevec2" )
                self.ProbePoints( "hess" )
               
                print "Assimilating data into polydata file:"
		readerWriter = ReadNRRDsWriteVTK()
		readerWriter.SetCIPBuildDirectory( self._cipBuildDirectory )
		readerWriter.AddFileNameArrayNamePair( self._temporaryDirectory + "pass1.nhdr",  "NA" )
		readerWriter.AddFileNameArrayNamePair( self._temporaryDirectory + "val.nrrd",    "val" )
		readerWriter.AddFileNameArrayNamePair( self._temporaryDirectory + "heval0.nrrd", "h0" )
		readerWriter.AddFileNameArrayNamePair( self._temporaryDirectory + "heval1.nrrd", "h1" )
		readerWriter.AddFileNameArrayNamePair( self._temporaryDirectory + "heval2.nrrd", "h2" )
		readerWriter.AddFileNameArrayNamePair( self._temporaryDirectory + "hmode.nrrd",  "hmode" )
		readerWriter.AddFileNameArrayNamePair( self._temporaryDirectory + "hevec0.nrrd", "hevec0" )
		readerWriter.AddFileNameArrayNamePair( self._temporaryDirectory + "hevec1.nrrd", "hevec1" )
		readerWriter.AddFileNameArrayNamePair( self._temporaryDirectory + "hevec2.nrrd", "hevec2" )
		readerWriter.AddFileNameArrayNamePair( self._temporaryDirectory + "hess.nrrd",   "hess" )
		readerWriter.SetOutputFileName( self._particlesFileName )
		readerWriter.Execute()

                if self._cleanTemporaryDirectory == True:
                        print "Cleaning tempoarary directory..."
                        tmpCommand = "rm " + self._temporaryDirectory + "ct-" + str(self._scale) + ".nrrd"
                        subprocess.call( tmpCommand, shell=True )

                        tmpCommand = "rm " + self._temporaryDirectory + "heval0.nrrd"
                        subprocess.call( tmpCommand, shell=True )

                        tmpCommand = "rm " + self._temporaryDirectory + "heval1.nrrd"
                        subprocess.call( tmpCommand, shell=True )

                        tmpCommand = "rm " + self._temporaryDirectory + "heval2.nrrd"
                        subprocess.call( tmpCommand, shell=True )

                        tmpCommand = "rm " + self._temporaryDirectory + "hmode.nrrd"
                        subprocess.call( tmpCommand, shell=True )

                        tmpCommand = "rm " + self._temporaryDirectory + "hevec0.nrrd"
                        subprocess.call( tmpCommand, shell=True )

                        tmpCommand = "rm " + self._temporaryDirectory + "hevec1.nrrd"
                        subprocess.call( tmpCommand, shell=True )

                        tmpCommand = "rm " + self._temporaryDirectory + "hevec2.nrrd"
                        subprocess.call( tmpCommand, shell=True )

                        tmpCommand = "rm " + self._temporaryDirectory + "val.nrrd"
                        subprocess.call( tmpCommand, shell=True )

                        tmpCommand = "rm " + self._temporaryDirectory + "pass1.nhdr"
                        subprocess.call( tmpCommand, shell=True )

                        tmpCommand = "rm " + self._temporaryDirectory + "pass1.raw"
                        subprocess.call( tmpCommand, shell=True )

                        tmpCommand = "rm " + self._temporaryDirectory + "lungMask.nrrd"
                        subprocess.call( tmpCommand, shell=True )

                        tmpCommand = "rm " + self._temporaryDirectory + "hess.nrrd"
                        subprocess.call( tmpCommand, shell=True )

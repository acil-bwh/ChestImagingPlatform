#!/usr/bin/python

#------------------------------------------------------------------------------------
# TODO: We will want to put some of the functionality into a base class from which 
# 'Generate[Airway,Vessel,Fissure]Particles' can inherit. 
#
# TODO: Create a separate class for probing points (post-processing unu calls)
#------------------------------------------------------------------------------------

import subprocess
from subprocess import PIPE
from ReadNRRDsWriteVTKModule import ReadNRRDsWriteVTKModule

class GenerateVesselParticles:
        def __init__( self ):                
                self._temporaryDirectory = "."
                self._inputFileName      = "NA"                        
                self._mask               = "NA"

                #       
                # Free parameters
                #
                self._theta          = 0.0  # TODO: Description
                self._liveThreshold  = -150 # TODO: Description
                self._seedThreshold  = -100 # TODO: Description
                self._hmodeThreshold = 0    # TODO: Rename ("hmode" is cryptic) and description
                self._maxIntensity   = 400  # TODO: Description
                
                #
                # Parameters for each pass. 
                #
                self._pass1Iterations = 10 # TODO: Description. Set to what?
                self._pass2Iterations = 20 # TODO: Description
                self._pass3Iterations = 70 # TODO: Description
                self._maxScale        = 6  # TODO: Description
                self._scaleSamples    = 10 # TODO: Description

        def SetInputFileName( self, fileName ):
                self._inputFileName = fileName

        def SetTemporaryDirectory( self, dir ):
                self._temporaryDirectory = dir

        def SetMask( self, mask ):
                self._mask = mask

        def ProbePoints( self, quantity ):
                Pass
                #tmpCommand = "gprobe -i " + self._temporaryDirectory + "/Deconvolved.nrrd + " -k scalar -k00 c4h -k11 c4hd -k22 c4hdd -pi " + \
                #self._temporaryDirectory + "outputPoints.nrrd + " -q val -o " + self._temporaryDirectory + "outputPoints.nrrd$inpoints-val.nrrd    -ssn $numScales -ssr 0 $maxScale -sso -ssf V-%03u-$numScalesTag.nrrd

                #subprocess.call( tmpCommand, shell=True )

        def Execute( self ):
                #
                # First perform deconvolution
                #
                tmpCommand = "unu pad -b pad -v -1000 -i " + self._inputFileName + " -min -0 -0 -0 \
                -max M M M | unu 3op clamp -850 - " + str(self._maxIntensity) + " | unu resample -s x1 x1 x1 \
                -k c4hai -t float -o " + self._temporaryDirectory + "/Deconvolved.nrrd"

                #subprocess.call( tmpCommand, shell=True )
		
                #
                # Execute the first pass
                #
                maskInfoParams = "spthr:M:val:0.5:1"
                output         = self._temporaryDirectory + "pass1.nrrd"

                self.ExecutePass( maskInfoParams, "false", "uni", 1, 0.7, 1, 1.7, 1.2, output, self._pass1Iterations )

                #
                # Execute the second pass
                #
                maskInfoParams = "spthr:M:val:-0.5:1"
                output         = self._temporaryDirectory + "pass2.nrrd"

                #self.ExecutePass( maskInfoParams, "true", "add", 0.0, 0.5, 1, 1.15, 2, output, self._pass2Iterations )

                #
                # Execute the third pass
                #
                maskInfoParams = "spthr:M:val:-0.5:1"
                output         = self._temporaryDirectory + "outputPoints.nrrd"

                #self.ExecutePass( maskInfoParams, "true", "add", 0.25, 0.25, 0.002, 1.15, 4, output, self._pass3Iterations )

                #
                # Now probe points 
                #
                self.ProbePoints( "val" )
                self.ProbePoints( "heval0" )
                self.ProbePoints( "heval1" )
                self.ProbePoints( "heval2" )
                self.ProbePoints( "hmode" )
                self.ProbePoints( "hevec0" )
                self.ProbePoints( "hevec1" )
                self.ProbePoints( "hevec2" )
                self.ProbePoints( "hess" )

        #
        # 'maskInfoParams':  Information parameters for mask
        #
        # 'efs':  Whether or not strength contributes to particle-image energy (bool)
        #
        # 'interParticleEnergy':  Inter-particle energy type
        #
        # 'alpha':  Blend between particle-image (alpha=0) and inter-particle (alpha=1) energies (double)
        #
        # 'beta':  When using Phi2 energy, blend between pure space repulsion (beta=0) and scale attraction (beta=1) (double)
        #
        # 'gamma':  Scaling factor on energy from strength (double)
        #
        # 'irad':  Particle radius in spatial domain (double)
        #
        # 'srad':  Particle radius in scale domain (double)
        #
        # 'output':  The pass output
        #
        # 'iterations':  Pass iterations
        #
        def ExecutePass( self, maskInfoParams, efs, interParticleEnergy, alpha, beta, gamma, irad, srad, output, iterations ):
                volParams = " -vol " + self._temporaryDirectory + "Deconvolved.nrrd:scalar:0-" + str(self._scaleSamples) + "-" + str(self._maxScale) + "-o:V " \
                + self._temporaryDirectory + "Deconvolved.nrrd:scalar:0-" + str(self._scaleSamples) + "-" + str(self._maxScale) + "-on:VSN " + self._mask + ":scalar:M"

                infoParams = " -info sthr:VSN:heval1:" + str(self._seedThreshold) + ":-1  h-c:V:val:0:-1  hgvec:V:gvec  hhess:V:hess tan1:V:hevec1 tan2:V:hevec2 lthr:VSN:heval1:" \
                 + str(self._liveThreshold) + ":-1 strn:VSN:heval1:0:-1 "

                energyParams = " -efs " + efs + " -enr qwell:0.7 -ens bparab:10,0.7,-0.00 -enw butter:10,0.7 -int " + interParticleEnergy + " -alpha " + str(alpha) + \
                " -beta " + str(beta) + " -irad " + str(irad) + " -srad " + str(srad) + " -theta " + str(self._theta)

                reconKernelParams = " -nave true -v 1 -pbm 0 -k00 c4h -k11 c4hd -k22 c4hdd -kssr hermite "

                optimizerParams = " -pcp 3 -edpcmin 0.1 -edmin 0.0000001 -eip 0.00001 -ppv 1 -jit 1 -ess 0.5 -nss 1 -oss 2 -step 1 -maxci 10 -rng 45 -bws 1.2 "

                tmpCommand = "puller -sscp " + self._temporaryDirectory + " -cbst true " + volParams + infoParams + " " + maskInfoParams + energyParams + reconKernelParams \
                + optimizerParams + " -o " + output + " -maxi " + str(iterations)
                
                subprocess.call( tmpCommand, shell=True )



# TODO: Below is a dump of the shell scripts that have been used to run vessel particles code. The goal
# is to migrate the functionality below to the python implementation above.

####################################################
# vesselParticlesMultiPass.sh
####################################################

##Free params
#set theta    =  0.0
#set livethr  = -150
#set seedthr  = -100
#set hmodethr =  0
#set maxI     =  400

##Params for each pass
#set iterPass1    = $numiter
#set iterPass2    = 20
#set iterPass3    = 70
#set maxScale     = 6
#set scaleSamples = 10

#set outP1 = p-pass1.nrrd
#set outP2 = p-pass2.nrrd
#set outP3 = $outpoints.nrrd

#set pad=0

#set infoParamsMask  = "spthr:M:val:0.5:1"
#set infoParamsMask2 = "spthr:M:val:-0.5:1"

#set reconKernelParams = '-nave true -v 1 -pbm 0 -k00 c4h -k11 c4hd -k22 c4hdd -kssr hermite'
#set optimizerParams   = '-pcp 3 -edpcmin 0.1 -edmin 0.0000001 -eip 0.00001 -ppv 1 -jit 1 -ess 0.5 -nss 1 -oss 2 -step 1 -maxci 10 -rng 45 -bws 1.2'

#set energyParamsP1         = "-efs false -enr qwell:0.7 -ens bparab:10,0.7,-0.00 -enw butter:10,0.7 -int uni -alpha 1    -beta 0.7               -irad 1.7  -srad 1.2 -theta $theta"
#set energyParamsP2         = "-efs true  -enr qwell:0.7 -ens bparab:10,0.7,-0.00 -enw butter:10,0.7 -int add -alpha 0.0  -beta 0.5               -irad 1.15 -srad 2   -theta $theta"
#set energyParamsP3         = "-efs true  -enr qwell:0.7 -ens bparab:10,0.7,-0.00 -enw butter:10,0.7 -int add -alpha 0.25 -beta 0.25 -gamma 0.002 -irad 1.15 -srad 4   -theta $theta"

##Pass 1:
#$puller -sscp ./ -cbst true \
# $volParams \
# $infoParams $infoParamsMask \
# $_firstPassEnergyParams \
# $reconKernelParams \
# $optimizerParams \
# -o $outP1 -maxi $iterPass1

##Pass 2:
#$puller -sscp ./ -cbst true \
#$volParams \
#$infoParams $infoParamsMask2 \
#$energyParamsP2 \
#$reconKernelParams \
#$optimizerParams \
#-pi $outP1 \
#-o $outP2 -maxi $iterPass2 

##Pass 3:
#$puller -sscp ./ -cbst true \
#$volParams \
#$infoParams $infoParamsMask2 \
#$energyParamsP3 \
#$reconKernelParams \
#$optimizerParams \
#-pi $outP2 \
#-o $outP3 -maxi $iterPass3 


##-covs ${outcov}
#endif
##Saving points to vtk
#echo "Sampling at particles points..."
#tcsh /projects/lmi/people/rjosest/COPD/Scripts/airwayParticles/postprocessingwithoutcov.sh ${case} $outP3 $maxScale $scaleSamples $tmpdir


#echo "Saving to vtk..."

###########################################################################
# postprocessingwithoutcov.sh
###########################################################################
#"Usage: postprocessingwithoutcov.sh <input image> <input points> <max scale> <num scales>"

#set invol=$1
#set inpoints=$2
#set maxScale=$3
#set numScales=$4

#cd $5

#if ( $numScales < 10 ) then
#  set numScalesTag="00$numScales"
#else
#  set numScalesTag="0$numScales"
#endif

##Probe hessian quantities
#gprobe -i $invol -k scalar -k00 c4h -k11 c4hd -k22 c4hdd -pi $inpoints -q val    -o $inpoints-val.nrrd    -ssn $numScales -ssr 0 $maxScale -sso -ssf V-%03u-$numScalesTag.nrrd
#gprobe -i $invol -k scalar -k00 c4h -k11 c4hd -k22 c4hdd -pi $inpoints -q heval0 -o $inpoints-h0.nrrd     -ssn $numScales -ssr 0 $maxScale -sso -ssf V-%03u-$numScalesTag.nrrd
#gprobe -i $invol -k scalar -k00 c4h -k11 c4hd -k22 c4hdd -pi $inpoints -q heval1 -o $inpoints-h1.nrrd     -ssn $numScales -ssr 0 $maxScale -sso -ssf V-%03u-$numScalesTag.nrrd
#gprobe -i $invol -k scalar -k00 c4h -k11 c4hd -k22 c4hdd -pi $inpoints -q heval2 -o $inpoints-h2.nrrd     -ssn $numScales -ssr 0 $maxScale -sso -ssf V-%03u-$numScalesTag.nrrd
#gprobe -i $invol -k scalar -k00 c4h -k11 c4hd -k22 c4hdd -pi $inpoints -q hmode  -o $inpoints-hmode.nrrd  -ssn $numScales -ssr 0 $maxScale -sso -ssf V-%03u-$numScalesTag.nrrd
#gprobe -i $invol -k scalar -k00 c4h -k11 c4hd -k22 c4hdd -pi $inpoints -q hevec2 -o $inpoints-hevec2.nrrd -ssn $numScales -ssr 0 $maxScale -sso -ssf V-%03u-$numScalesTag.nrrd
#gprobe -i $invol -k scalar -k00 c4h -k11 c4hd -k22 c4hdd -pi $inpoints -q hevec1 -o $inpoints-hevec1.nrrd -ssn $numScales -ssr 0 $maxScale -sso -ssf V-%03u-$numScalesTag.nrrd
#gprobe -i $invol -k scalar -k00 c4h -k11 c4hd -k22 c4hdd -pi $inpoints -q hevec0 -o $inpoints-hevec0.nrrd -ssn $numScales -ssr 0 $maxScale -sso -ssf V-%03u-$numScalesTag.nrrd
#gprobe -i $invol -k scalar -k00 c4h -k11 c4hd -k22 c4hdd -pi $inpoints -q hess   -o $inpoints-hess.nrrd   -ssn $numScales -ssr 0 $maxScale -sso -ssf V-%03u-$numScalesTag.nrrd

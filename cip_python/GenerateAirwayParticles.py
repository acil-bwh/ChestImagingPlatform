#!/usr/bin/python

import os
import pdb
#import subprocess
#from subprocess import PIPE
#from ReadNRRDsWriteVTKModule import ReadNRRDsWriteVTKModule
from cip_python.GenerateParticles import GenerateParticles

class GenerateAirwayParticles(GenerateParticles):
    def __init__( self ):
        GenerateParticles.__init__( self )
        self.SetFeatureTypeToValleyLine()
        self._maxIntensity = -600
        self._minIntensity = -1100

    def Execute( self ):
        #Pre-processing
        if self._downSamplingRate > 1:
            downsampledVolume = os.path.join(self._temporaryDirectory,"ct-down.nrrd")
            self.DownSampling(self._inputFileName,downsampledVolume,"cubic:0,0.5")
            if self._useMask == 1:
            	downsampledMask = os.path.join(self._temporaryDirectory,"mask-down.nrrd")
            	self.DownSampling(self._maskFileName,downsampledMask,"cheap")
            	self._maskFileName = downsampledMask
            
        else:
            downsampledVolume = self._inputFileName

        print "2"
        deconvolvedVolume = os.path.join(self._temporaryDirectory,"ct-deconv.nrrd")
        self.Deconvolution(downsampledVolume,deconvolvedVolume)
        print "finished deconvolution\n"
        print "loc1\n"
        #Setting member variables that will not change
        self._inputVolume = deconvolvedVolume
        print "loc2\n"
        
        #Temporate nrrd particles points
        outputParticles = os.path.join(self._temporaryDirectory, "pass%d.nrrd")
        print "loc3\n"
        #Pass 1
        #Init params
        self._useStrength = False
        self._interParticleEnergyType = "uni"
        self._initializationMode = "PerVoxel"
        self._ppv = 1
        self._nss=2

        self._liveThreshold = 50
        self._seedThreshold = 40 #Threshold on strengh feature

        #Energy
        self._interParticleEneryType = "uni"
        self._beta  = 0.7
        self._alpha = 1.0
        self._iterations = 10

        #Build parameters and run
        print "resetting param groups\n"
        self.ResetParamGroups()
        print "building param groups\n"
        self.BuildParamGroups()
        print "Starting pass 1\n"
        self.ExecutePass(outputParticles % 1)
        print "Finished pass 1\n"

        #Pass 2
        #Init params
        self._initializationMode = "Particles"
        self._inputNRRDParticlesFileName = outputParticles % 1
        self._useMask = 0

        #Energy
        self._interParticleEnergyType = "add"
        self._alpha = 0
        self._beta = 0.5
        self._irad = 1.15
        self._srad = 4
        self._useStrength = True

        self._iterations = 20

        #Build parameters and run
        self.ResetParamGroups()
        self.BuildParamGroups()
        print "starting pass 2\n"
        self.ExecutePass(outputParticles % 2)
        print "finished pass 2\n"

        #Pass 3
        self._initializationMode = "Particles"
        self._inputNRRDParticlesFileName = outputParticles % 2
        self._useMask = 0

        #Energy
        self._interParticleEnergyType = "add"
        self._alpha = 0.5
        self._beta = 0.5
        self._gamma = 0.000002
        self._irad = 1.15
        self._srad = 4
        self._useStrength = True

        self._iterations = 50

        #Build parameters and run
        self.ResetParamGroups()
        self.BuildParamGroups()
        print "starting pass 3\n"
        self.ExecutePass(outputParticles % 3)
        print "finished pass 3\n"

        #Probe quantities and save to VTK
        print "about to probe\n"
        self.ProbeAllQuantities(self._inputVolume, outputParticles % 3)
        print "finished probing\n"
        print "about to save to vtk\n"
        self.SaveParticlesToVTK(outputParticles % 3)
        print "finished saving\n"

########################################################
# runairwaypipeline.sh
########################################################
#set dir=$1
#set caseID=$2
#set tmpdir=$3
#set runLeftLung=$4
#set runRightLung=$5
#set execdir=/projects/lmi/people/rjosest/COPD/Scripts/airwayParticles
#set caseT=$dir/$caseID.nhdr
#set lungmask=$dir/${caseID}_partialLungLabelMap.nhdr
#
#set caseL=$tmpdir/L/imL.nhdr
#set lungmaskL=$tmpdir/L/imL_partialLungLabelMap.nhdr
#
#
#set deconvL=$tmpdir/L/deconvL.nhdr
#set airwaymaskL=$tmpdir/L/airwaymaskL.nhdr
#
#set slicer=/projects/lmi/people/rjosest/src/devel/slicer2/slicer2-linux-x86
#
#set iterPhase1=80
#set iterPhase2=10
#set iterPhase3=70
#
#$slicer $execdir/cropLungRegion.tcl $caseT $lungmask L $caseL $lungmaskL
#$slicer $execdir/generateAirwayMaskFromVessels.tcl $caseL $lungmaskL $airwaymaskL
#tclsh $execdir/replaceOriging.tcl $caseL $airwaymaskL
#
#echo "Deconvolution $caseT left lung"
#$execdir/preprocessing.sh $caseL $deconvL
#tcsh $execdir/airwayphase1.sh $deconvL $airwaymaskL $iterPhase1 $tmpdir/L/pL_phase1.nhdr $tmpdir/L/pL_cov.nhdr $tmpdir/L
#tcsh $execdir/airwayphase2.sh $deconvL $tmpdir/L/pL_phase1.nhdr $iterPhase2 $tmpdir/L/pL_phase2.nhdr $tmpdir/L/pL_cov.nhdr $tmpdir/L
#tcsh $execdir/airwayphase3.sh $deconvL $tmpdir/L/pL_phase2.nhdr $iterPhase3 $tmpdir/L/pL_phase3.nhdr $tmpdir/L/pL_phase3.nhdr-cov.nrrd $tmpdir/L
#tcsh $execdir/postprocessing.sh $deconvL $tmpdir/L/pL_phase3.nhdr $tmpdir/L/pL_phase3.nhdr-cov.nrrd $tmpdir/L/pL_final.nhdr $tmpdir/L
#
#set inpoints=$tmpdir/L/pL_phase3.nhdr
#set outpoints=$tmpdir/${caseID}_leftAirwayParticles.vtk
#
#
#if ( $runRightLung == 1 ) then
#echo "${caseT}: Processing Right Lung..."
#$slicer $execdir/cropLungRegion.tcl $caseT $lungmask R $caseR $lungmaskR
#$slicer $execdir/generateAirwayMaskFromVessels.tcl $caseR $lungmaskR $airwaymaskR
#tclsh $execdir/replaceOriging.tcl $caseR $airwaymaskR
#
#echo "Deconvolution $caseT right lung"
#$execdir/preprocessing.sh $caseR $deconvR
#echo "Scale-space particles $caseT right lung"
#tcsh $execdir/airwayphase1.sh $deconvR $airwaymaskR $iterPhase1 $tmpdir/R/pR_phase1.nhdr $tmpdir/R/pR_cov.nhdr $tmpdir/R
#tcsh $execdir/airwayphase2.sh $deconvR $tmpdir/R/pR_phase1.nhdr $iterPhase2 $tmpdir/R/pR_phase2.nhdr $tmpdir/R/pR_cov.nhdr $tmpdir/R
#tcsh $execdir/airwayphase3.sh $deconvR $tmpdir/R/pR_phase2.nhdr $iterPhase3 $tmpdir/R/pR_phase3.nhdr $tmpdir/R/pR_phase3.nhdr-cov.nrrd $tmpdir/R
#echo "Postprocessing"
#tcsh $execdir/postprocessing.sh $deconvR $tmpdir/R/pR_phase3.nhdr $tmpdir/R/pR_phase3.nhdr-cov.nrrd $tmpdir/R/pR_final.nhdr $tmpdir/R
#
##echo "$caseT Right: Converting to VTK..."
#set inpoints=$tmpdir/R/pR_phase3.nhdr
#set outpoints=$tmpdir/${caseID}_rightAirwayParticles.vtk
#
#
################################################################
## preprocessing.sh
################################################################
##preprocessing.sh <input image> <output image> <tmpdir>
#
#set invol=$1
#set out=$2
#
#unu quantize -i $invol -b 16 -min -1024 -max -700 \
# | unu quantize -b 16 -min -10000 -max 75536 \
# | unu resample -s x1 x1 x1 -k c4hai -o ${out}
#
#############################################################################
## airwayphase1
#############################################################################
##airwayphase1.sh <input image> <input mask> <numiter> <output points> <tmpdir>
#
#set case=$1
#set mask=$2
#set numiter=$3
#set outpoints=$4
#set outcov=$5
#cd $6
#
#setenv DEFT_HOME /home/rjosest/projects/src/devel/Deft
#
#/home/rjosest/projects/src/devel/Deft/src/puller -sscp ./ -cbst true \
# -vol ${case}:scalar:0-5-6-o:V \
#      ${case}:scalar:0-5-6-on:VSN \
#      ${mask}:scalar:M \
# -info sthr:VSN:heval1:10000:1  h-c:V:val:0:1  hgvec:V:gvec  hhess:V:hess  \
#       tan1:V:hevec0 tan2:V:hevec1  \
#       lthr:VSN:heval1:10000:1 \
#       strn:VSN:heval1:0:1 \
#       spthr:M:val:0.5:1 \
#\
# -efs false -enr qwell:0.7 -ens bparab:10,0.7,-0.00 -enw butter:10,0.7 \
#   -int uni -alpha 1 -beta 0.7 -irad 1.7 -srad 1.2 \
#\
#-nave true -v 0 \
#  -k00 c4h -k11 c4hd -k22 c4hdd -kssr hermite \
#  -pcp 5 -edpcmin 0.1 -edmin 0.0000001 -eip 0.00001 \
#  -ppv -3 -jit 1 -ess 0.5 -oss 2 -step 1 -maxci 10  \
#  -rng 45 -pbm 0 \
#  -o ${outpoints} -maxi $numiter -covs ${outcov}
#
#exit
#
#
##################################################################################
## airwayphase2
##################################################################################
##!/bin/tcsh
##airwayphase2.sh <input image> <input points> <numiter> <output points> <tmpdir>
##set case=10013P_EXP_STD_BWH_COPD
#
#if ( $#argv  != 6 ) then
#  echo "Usage: airwayphase2.sh <input image> <input points> <numiter> <output points> <output cov> <tmpdir>"
#  exit
#endif
#
#set case=$1
#set inpoints=$2
#set numiter=$3
#set outpoints=$4
#set outcov=$5
#cd $6
#
#setenv DEFT_HOME /home/rjosest/projects/src/devel/Deft
#
#/home/rjosest/projects/src/devel/Deft/src/puller -sscp ./ -cbst true \
# -vol ${case}:scalar:0-5-6-o:V \
#      ${case}:scalar:0-5-6-on:VSN \
# -info sthr:VSN:heval1:10000:1  h-c:V:val:0:1  hgvec:V:gvec  hhess:V:hess  \
#       tan1:V:hevec0 tan2:V:hevec1  \
#       lthr:VSN:heval1:10000:1 \
#       strn:VSN:heval1:0:1 \
#\
# -efs true -enr qwell:0.7 -ens bparab:10,0.7,-0.00 -enw butter:10,0.7 \
#   -int add -alpha 0.0 -beta 0.5 -irad 1.15 -srad 4 \
#\
#-nave true -v 0 \
#  -k00 c4h -k11 c4hd -k22 c4hdd -kssr hermite \
#  -pcp 5 -edpcmin 0.1 -edmin 0.0000001 -eip 0.00001 \
#  -np 1000 -ess 0.5 -oss 2 -step 1 -maxci 10  \
#  -rng 45 -pbm 0 \
#  -pi ${inpoints} -o ${outpoints} -maxi $numiter -covs ${outcov}
#
#exit
#
#
###########################################################################################
## airwayphase3
###########################################################################################
##!/bin/tcsh
##airwayphase1.sh <input image> <input points> <numiter> <output points> <tmpdir>
##set case=10013P_EXP_STD_BWH_COPD
#
#
#if ( $#argv  != 6 ) then
#  echo "Usage: airwayphase3.sh <input image> <input points> <numiter> <output points> <output cov> <tmpdir>"
#  exit
#endif
#
#set case=$1
#set inpoints=$2
#set numiter=$3
#set outpoints=$4
#set outcov=$5
#
#cd $6
#
##set alpha=0.25
#set alpha=0.5
##set beta=0.25
#set beta=0.5
#
#setenv DEFT_HOME /home/rjosest/projects/src/devel/Deft
#
#/home/rjosest/projects/src/devel/Deft/src/puller -sscp ./ -cbst true \
# -vol ${case}:scalar:0-5-6-o:V \
#      ${case}:scalar:0-5-6-on:VSN \
#-info sthr:VSN:heval1:9000:1  h-c:V:val:0:1  hgvec:V:gvec  hhess:V:hess  \
#       tan1:V:hevec0 tan2:V:hevec1  \
#       lthr:VSN:heval1:9000:1 \
#       strn:VSN:heval1:0:1 \
#       qual:VSN:val:0:-1 \
#\
# -efs true -enr qwell:0.7 -ens bparab:10,0.7,-0.00 -enw butter:10,0.7 \
#   -int add -alpha $alpha -beta $beta -gamma 0.000002 -irad 1.15 -srad 4 \
#\
#-nave true -v 0 \
#  -k00 c4h -k11 c4hd -k22 c4hdd -kssr hermite \
#  -pcp 5 -edpcmin 0.1 -edmin 0.0000001 -eip 0.00001 \
#  -np 1000 -ess 0.5 -oss 2 -step 1 -maxci 10  \
#  -rng 45 -pbm 0 \
#  -pi ${inpoints} -o ${outpoints} -maxi $numiter -covs ${outcov}
#
#exit
#
#
############################################################################
## postprocessing
############################################################################
##!/bin/tcsh
##preprocessing.sh <input image> <output image> <tmpdir>
#
#if ( $#argv  != 5 ) then
#  echo "Usage: postprocessing.sh <input image> <input points> <input cov> <output points> <tmpdir>"
#  exit
#endif
#
#set invol=$1
#set inpoints=$2
#set incov=$3
#set outpoints=$4
#
#cd $5
#
#echo Hola
#
##Probe hessian quantities
#gprobe -i $invol -k scalar -k00 c4h -k11 c4hd -k22 c4hdd -pi $inpoints -q val -o $inpoints-val.nrrd -ssn 5 -ssr 0 6 -sso -ssf V-%03u-005.nrrd
#gprobe -i $invol -k scalar -k00 c4h -k11 c4hd -k22 c4hdd -pi $inpoints -q heval0 -o $inpoints-h0.nrrd -ssn 5 -ssr 0 6 -sso -ssf V-%03u-005.nrrd
#gprobe -i $invol -k scalar -k00 c4h -k11 c4hd -k22 c4hdd -pi $inpoints -q heval1 -o $inpoints-h1.nrrd -ssn 5 -ssr 0 6 -sso -ssf V-%03u-005.nrrd
#gprobe -i $invol -k scalar -k00 c4h -k11 c4hd -k22 c4hdd -pi $inpoints -q heval2 -o $inpoints-h2.nrrd -ssn 5 -ssr 0 6 -sso -ssf V-%03u-005.nrrd
#gprobe -i $invol -k scalar -k00 c4h -k11 c4hd -k22 c4hdd -pi $inpoints -q hmode -o $inpoints-hmode.nrrd -ssn 5 -ssr 0 6 -sso -ssf V-%03u-005.nrrd
#gprobe -i $invol -k scalar -k00 c4h -k11 c4hd -k22 c4hdd -pi $inpoints -q hevec2 -o $inpoints-hevec2.nrrd -ssn 5 -ssr 0 6 -sso -ssf V-%03u-005.nrrd
#gprobe -i $invol -k scalar -k00 c4h -k11 c4hd -k22 c4hdd -pi $inpoints -q hevec1 -o $inpoints-hevec1.nrrd -ssn 5 -ssr 0 6 -sso -ssf V-%03u-005.nrrd
#gprobe -i $invol -k scalar -k00 c4h -k11 c4hd -k22 c4hdd -pi $inpoints -q hevec0 -o $inpoints-hevec0.nrrd -ssn 5 -ssr 0 6 -sso -ssf V-%03u-005.nrrd
#gprobe -i $invol -k scalar -k00 c4h -k11 c4hd -k22 c4hdd -pi $inpoints -q hess -o $inpoints-hess.nrrd -ssn 5 -ssr 0 6 -sso -ssf V-%03u-005.nrrd
#
##gprobe -i $invol -k scalar -k00 cubic:1,0 -k11 cubicd:1,0 -k22 cubicdd:1,0 -pi $inpoints -q val -o $inpoints-val.nrrd -ssn 5 -ssr 0 6 -ssf V-%03u-005.nrrd
##gprobe -i $invol -k scalar -k00 cubic:1,0 -k11 cubicd:1,0d -k22 cubicdd:1,0 -pi $inpoints -q heval0 -o $inpoints-h0.nrrd -ssn 5 -ssr 0 6 -ssf V-%03u-005.nrrd
##gprobe -i $invol -k scalar -k00 cubic:1,0 -k11 cubicd:1,0 -k22 cubicdd:1,0 -pi $inpoints -q heval1 -o $inpoints-h1.nrrd -ssn 5 -ssr 0 6 -ssf V-%03u-005.nrrd
##gprobe -i $invol -k scalar -k00 cubic:1,0 -k11 cubicd:1,0 -k22 cubicdd:1,0 -pi $inpoints -q heval2 -o $inpoints-h2.nrrd -ssn 5 -ssr 0 6 -ssf V-%03u-005.nrrd
##gprobe -i $invol -k scalar -k00 cubic:1,0 -k11 cubicd:1,0 -k22 cubicdd:1,0 -pi $inpoints -q hmode -o $inpoints-hmode.nrrd -ssn 5 -ssr 0 6 -ssf V-%03u-005.nrrd
##gprobe -i $invol -k scalar -k00 cubic:1,0 -k11 cubicd:1,0 -k22 cubicdd:1,0 -pi $inpoints -q hevec2 -o $inpoints-hevec2.nrrd -ssn 5 -ssr 0 6 -ssf V-%03u-005.nrrd
##gprobe -i $invol -k scalar -k00 cubic:1,0 -k11 cubicd:1,0 -k22 cubicdd:1,0 -pi $inpoints -q hevec1 -o $inpoints-hevec1.nrrd -ssn 5 -ssr 0 6 -ssf V-%03u-005.nrrd
##gprobe -i $invol -k scalar -k00 cubic:1,0 -k11 cubicd:1,0 -k22 cubicdd:1,0 -pi $inpoints -q hevec0 -o $inpoints-hevec0.nrrd -ssn 5 -ssr 0 6 -ssf V-%03u-005.nrrd
#
#
#
##gprobe -i deconv.nhdr -k scalar -k00 cubic:1,0 -k11 cubicd:1,0 -k22 cubicdd:1,0 -pi p_phase3.ndhr -q val -o aa.nrrd -ssn 5 -ssr 0 6 -ssf V-%%03u-%03u.nrrd
##Extract covariance information
#unu axinsert -a 2 -i $incov | unu axinsert -a 3 -i - | tend anvol -a cl1 -i - -o $inpoints-covcl.nrrd
#unu axinsert -a 2 -i $incov | unu axinsert -a 3 -i - | tend evec -c 2 -i - | unu axmerge -i - -a 1 2 -o $inpoints-covevec2.nrrd
#unu axinsert -a 2 -i $incov | unu axinsert -a 3 -i - | tend evec -c 1 -i - | unu axmerge -i - -a 1 2 -o $inpoints-covevec1.nrrd
#unu axinsert -a 2 -i $incov | unu axinsert -a 3 -i - | tend evec -c 0 -i - | unu axmerge -i - -a 1 2 -o $inpoints-covevec0.nrrd
#unu axinsert -a 2 -i $incov | unu axinsert -a 3 -i - | tend eval -c 0 -i - | unu permute -i - -p 1 2 0 | unu axmerge -a 0 -o $inpoints-coveval0.nrrd
#unu axinsert -a 2 -i $incov | unu axinsert -a 3 -i - | tend eval -c 1 -i - | unu permute -i - -p 1 2 0 | unu axmerge -a 0 -o $inpoints-coveval1.nrrd
#unu axinsert -a 2 -i $incov | unu axinsert -a 3 -i - | tend eval -c 2 -i - | unu permute -i - -p 1 2 0 | unu axmerge -a 0 -o $inpoints-coveval2.nrrd
#
#
##Run matlab postprocessing script
##/projects/lmi/software/Matlab2008b64bit/bin/matlab -nodesktop -nojvm -nosplash -r "addpath /home/rjosest/projects/src/devel/teem/src/matlab/; addpath /projects/lmi/people/rjosest/COPD/Scripts/airwayParticles; filterPoints('$inpoints','$outpoints'); exit;" &
#set cmd="matlab64 -nodesktop -nojvm -nosplash -r \"addpath /home/rjosest/projects/src/devel/teem/src/matlab/; filterPoints($inpoints,$outpoints)\""

#tclsh /projects/lmi/people/rjosest/COPD/Scripts/airwayParticles/filterPoints.tcl $inpoints $outpoints

#echo $cmd

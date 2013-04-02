#!/usr/bin/python


from cipPython import GenerateVesselParticles
#from GenerateFissureParticlesModule import GenerateFissureParticlesModule

particlesGenerator = GenerateFissureParticlesModule()
particlesGenerator.SetTemporaryDirectory( "/spl/tmp/jross/" )
particlesGenerator.SetInput( "/net/th914_nas.bwh.harvard.edu/mnt/array1/share/Processed/COPDGene/11313G/11313G_INSP_STD_TXS_COPD/11313G_INSP_STD_TXS_COPD.nhdr" )
particlesGenerator.SetMask(  "/net/th914_nas.bwh.harvard.edu/mnt/array1/share/Processed/COPDGene/11313G/11313G_INSP_STD_TXS_COPD/11313G_INSP_STD_TXS_COPD_leftLungRightLung.nhdr" )
particlesGenerator.SetCIPBuildDirectory( "/projects/lmi/people/jross/Downloads/cip/trunk/Build/" )
particlesGenerator.SetOutputFileName( "/spl/tmp/jross/foo.vtk" )
particlesGenerator.SetCleanTemporaryDirectory( False )
particlesGenerator.Execute()


#!/usr/bin/python

import sys
import string
from optparse import OptionParser

parser = OptionParser()
parser.add_option("-f", dest="caseListFile")
parser.add_option("-s", type="int", dest="startCase", default=1)
parser.add_option("-e", type="int", dest="endCase", default=sys.maxint)
parser.add_option("--cipPython", dest="cipPythonDirectory")
parser.add_option("--cipBuildDir", dest="cipBuildDirectory")
parser.add_option("--tmpDir", dest="tmpDirectory")
parser.add_option("--dataDir", dest="dataDirectory")
parser.add_option("--copdGeneDataDir", dest="copdGeneDataDir", default="NA")

(options, args) = parser.parse_args()

sys.path.append(options.cipPythonDirectory)

from cipPython.GenerateAirwayParticles import GenerateAirwayParticles

caseListFile = open(options.caseListFile,'r')

inc=-1
for case in caseListFile:
    inc = inc+1
    if inc >= options.startCase and inc <= options.endCase:
        print "---------------------------------"
        print case

        particlesGenerator = GenerateAirwayParticles()
        particlesGenerator.SetCleanTemporaryDirectory( False )
        particlesGenerator.SetCIPBuildDirectory( options.cipBuildDirectory )
        particlesGenerator.SetTemporaryDirectory( options.tmpDirectory )
        #particlesGenerator.SetInput( options.dataDirectory + case.rstrip() + '/' + case.rstrip() + '_oriented.nhdr' )
        #particlesGenerator.SetMask(  options.dataDirectory + case.rstrip() + '/' + case.rstrip() + '_oriented_leftLungRightLung.nhdr' )
        #particlesGenerator.SetOutputFileName( options.dataDirectory + case.rstrip() + '/' + case.rstrip() + '_oriented_fissureParticles.vtk' )
        particlesGenerator.SetInput( "/Users/jross/Projects/Data/COPDGene/11310A/11310A_INSP_STD_JHU_COPD/11310A_INSP_STD_JHU_COPD_cropped.nhdr" )
        particlesGenerator.SetMask( "/Users/jross/Projects/Data/COPDGene/11310A/11310A_INSP_STD_JHU_COPD/11310A_INSP_STD_JHU_COPD_airwayCropped.nhdr"  )
        particlesGenerator.SetOutputFileName( "/Users/jross/Projects/Data/COPDGene/11310A/11310A_INSP_STD_JHU_COPD/11310A_INSP_STD_JHU_COPD_test.vtk" )
        particlesGenerator.Execute()
        

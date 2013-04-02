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

(options, args) = parser.parse_args()

#sys.path.append("/Users/jross/Downloads/cip/trunk/")
sys.path.append(options.cipPythonDirectory)

from cipPython.GenerateFissureParticles import GenerateFissureParticles

caseListFile = open(options.caseListFile,'r')

inc=0
for case in caseListFile:
    inc = inc+1
    if inc >= options.startCase and inc <= options.endCase:
        print "---------------------------------"
        print case

        particlesGenerator = GenerateFissureParticles()
        particlesGenerator.SetTemporaryDirectory( options.tmpDirectory )
        particlesGenerator.SetInput( options.dataDirectory + case.rstrip() + '/' + case.rstrip() + '_oriented.nhdr' )
        #particlesGenerator.SetMask(  options.dataDirectory + case.rstrip() + '/' + case.rstrip() + '_oriented_interactiveLobeSegmentationFissureMask.nhdr' )
        particlesGenerator.SetMask(  options.dataDirectory + case.rstrip() + '/' + case.rstrip() + '_oriented_leftLungRightLung.nhdr' )
        particlesGenerator.SetCIPBuildDirectory( options.cipBuildDirectory )
        particlesGenerator.SetOutputFileName( options.dataDirectory + case.rstrip() + '/' + case.rstrip() + '_oriented_fissureParticles.vtk' )
        particlesGenerator.SetCleanTemporaryDirectory( False )
        particlesGenerator.Execute()


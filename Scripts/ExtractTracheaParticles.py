#!/usr/bin/python

import sys
import string
import subprocess
import os

from optparse import OptionParser

parser = OptionParser()
parser.add_option("-c", dest="caseId")
parser.add_option("--cipPython", dest="cipPythonDirectory")
parser.add_option("--cipBuildDir", dest="cipBuildDirectory")
parser.add_option("--tmpDir", dest="tmpDirectory")
parser.add_option("--dataDir", dest="dataDirectory")
parser.add_option("--cleanCache", action="store_true", dest="cleanCache", default="False")

(options, args) = parser.parse_args()

sys.path.append(options.cipPythonDirectory)
from cipPython import GenerateAirwayParticles
from cipPython import GenerateFeatureStrengthMap

caseId = options.caseId

#Extract Trachea
plFileName = os.path.join(options.dataDirectory,caseId + "_partialLungLabelMap.nhdr")
ctFileName = os.path.join(options.dataDirectory,caseId + ".nhdr")
plFileNameTrachea = os.path.join(options.tmpDirectory,caseId + "_trachea_partialLungLabelMap.nhdr")
particlesFileNameTrachea = os.path.join(options.dataDirectory,caseId + "_tracheaAndMainBronchiParticles.vtk")

tmpCommand ="ExtractChestLabelMap -t 2 -i %(lm-in)s -o %(lm-out)s; unu 2op gt %(lm-out)s 0.5 -o %(lm-out)s"

tmpCommand = tmpCommand % {'lm-in':plFileName,'lm-out':plFileNameTrachea}
tmpCommand = os.path.join(options.cipBuildDirectory,"bin",tmpCommand)
subprocess.call( tmpCommand, shell=True )

particlesGenerator = GenerateAirwayParticles()
particlesGenerator.SetCleanTemporaryDirectory( options.cleanCache )
particlesGenerator.SetCIPBuildDirectory( options.cipBuildDirectory )
particlesGenerator.SetTemporaryDirectory( options.tmpDirectory )
particlesGenerator.SetInputFileName( ctFileName )
particlesGenerator.SetMaskFileName( plFileNameTrachea  )

particlesGenerator._downSamplingRate=2
particlesGenerator._maxScale = 5
particlesGenerator._scaleSamples = 5
particlesGenerator._liveThreshold = 110
particlesGenerator._seedThreshold = 90
particlesGenerator._maxIntensity = -200
particlesGenerator._minIntensity = -1100
particlesGenerator._ppv = -5
particlesGenerator.SetOutputFileName( particlesFileNameTrachea )

particlesGenerator.Execute()

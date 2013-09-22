#!/usr/bin/python

import sys
import string
import subprocess
import os

from optparse import OptionParser

parser = OptionParser()
parser.add_option("-c", dest="caseId")
parser.add_option("--cipPython", dest="cipPythonDirectory")
parser.add_option("--tmpDir", dest="tmpDirectory")
parser.add_option("--dataDir", dest="dataDirectory")
parser.add_option("--cleanCache", action="store_true", dest="cleanCache", default="False")
parser.add_option("-r",dest="region")

(options, args) = parser.parse_args()

sys.path.append(options.cipPythonDirectory)
from cip_python import GenerateAirwayParticles
from cip_python import GenerateFeatureStrengthMap
from cip_python.vessel_particles import VesselParticles

caseId = options.caseId

#region = [2,3]
#region=[2]
#regionTag = ['right','left']
region = [int(i) for i in str.split(options.region,',')]

regionTag = dict()
regionTag[2]='right'
regionTag[3]='left'

#Check required tools path enviroment variables for tools path
toolsPaths = ['CIP_PATH','TEEM_PATH','ITKTOOLS_PATH'];
path=dict()
for path_name in toolsPaths:
    path[path_name]=os.environ.get(path_name,False)
    if path[path_name] == False:
        print path_name + " environment variable is not set"
        exit()


plFileName = os.path.join(options.dataDirectory,caseId + "_partialLungLabelMap.nhdr")
ctFileName = os.path.join(options.dataDirectory,caseId + ".nhdr")

minFeatureStrength = 60


for ii in region:
    tmpDir = os.path.join(options.tmpDirectory,regionTag[ii])
    if os.path.exists(tmpDir) == False:
        os.mkdir(tmpDir)

    # Define FileNames that will be used
    plFileNameRegion= os.path.join(tmpDir,caseId + "_" + regionTag[ii] + "_partialLungLabelMap.nhdr")
    ctFileNameRegion= os.path.join(tmpDir,caseId + "_" + regionTag[ii] + ".nhdr")
    featureMapFileNameRegion = os.path.join(tmpDir,caseId + "_" + regionTag[ii] + "_featureMap.nhdr")
    maskFileNameRegion = os.path.join(tmpDir,caseId + "_" + regionTag[ii] + "_mask.nhdr")
    particlesFileNameRegion = os.path.join(options.dataDirectory,caseId + "_" + regionTag[ii] + "VesselParticles.vtk")

    #Create SubVolume Region
    tmpCommand ="CropLung -r %(region)d -m 1 -v 0 -i %(ct-in)s --plf %(lm-in)s -o %(ct-out)s --opl %(lm-out)s"
    tmpCommand = tmpCommand % {'region':ii,'ct-in':ctFileName,'lm-in':plFileName,'ct-out':ctFileNameRegion,'lm-out':plFileNameRegion}
    tmpCommand = os.path.join(path['CIP_PATH'],tmpCommand)
    print tmpCommand
# subprocess.call( tmpCommand, shell=True )

    #Extract Lung Region + Distance map to peel lung
    tmpCommand ="ExtractChestLabelMap -r %(region)d -i %(lm-in)s -o %(lm-out)s"
    tmpCommand = tmpCommand % {'region':ii,'lm-in':plFileNameRegion,'lm-out':plFileNameRegion}
    tmpCommand = os.path.join(path['CIP_PATH'],tmpCommand)
    print tmpCommand
#subprocess.call( tmpCommand, shell=True )

    tmpCommand ="unu 2op gt %(lm-in)s 0.5 -o %(lm-out)s"
    tmpCommand = tmpCommand % {'lm-in':plFileNameRegion,'lm-out':plFileNameRegion}
    print tmpCommand
###subprocess.call( tmpCommand, shell=True )

    tmpCommand ="pxdistancetransform -in %(lm-in)s -out %(distance-map)s"
    tmpCommand = tmpCommand % {'lm-in':plFileNameRegion,'distance-map':plFileNameRegion}
    tmpCommand = os.path.join(path['ITKTOOLS_PATH'],tmpCommand)
    print tmpCommand
#subprocess.call( tmpCommand, shell=True )

    tmpCommand ="unu 2op lt %(distance-map)s -2 -t short -o %(lm-out)s"
    tmpCommand = tmpCommand % {'distance-map':plFileNameRegion,'lm-out':plFileNameRegion}
    print tmpCommand
#subprocess.call( tmpCommand, shell=True )


    if False:

        # Create Feature Strength Map for the Region
        strengthMap = FeatureStrengthMap("ridge_line",ctFileNameRegion,featureMapFileNameRegion,tmpDir)
        strengthMap._clean_tmp_dir=False
        strengthMap._max_scale = 4
        strengthMap._scale_samples = 5
        strengthMap._max_feature_strength = 10000
        strengthMap._probe_scales = [0.75,1,1.75,3]

        strengthMap.execute()

        # Threshold Feature Strength Map
        tmpCommand ="unu 2op gt %(input)s %(min-strength)f -t short -o %(output)s; unu 2op gt %(lm-input)s 0.5 | unu 2op x - %(output)s -o %(output)s"
        tmpCommand = tmpCommand % {'input':featureMapFileNameRegion,'min-strength':minFeatureStrength,'lm-input':plFileNameRegion,'output':maskFileNameRegion}
        print tmpCommand
        subprocess.call( tmpCommand, shell=True )

# Compute Frangi

    tmpCommand = "pxenhancement -in %(in)s -m FrangiVesselness -std 1 3 3 -ssm 0 -alpha 0.5 -beta 0.5 -C 300 -out %(out)s"
    tmpCommand = tmpCommand % {'in':ctFileNameRegion,'out':featureMapFileNameRegion}


    tmpCommand  = os.path.join(path['ITKTOOLS_PATH'],tmpCommand)
    print tmpCommand
#subprocess.call( tmpCommand, shell=True )
    
    
    #Hist equalization, threshold Feature strength and masking
    tmpCommand = "unu 2op x %(feat)s %(mask)s -t float | unu heq -b 10000 -a 0.5 | unu 2op gt - 0.6  | unu convert -t short -o %(out)s"
    tmpCommand = tmpCommand % {'feat':featureMapFileNameRegion,'mask':plFileNameRegion,'out':maskFileNameRegion}
    
    print tmpCommand
    subprocess.call( tmpCommand , shell=True)


    # Airway Particles For the Region
    particlesGenerator = VesselParticles(ctFileNameRegion,particlesFileNameRegion,tmpDir,maskFileNameRegion)
    particlesGenerator._clean_tmp_dir=options.cleanCache
    particlesGenerator.execute()

